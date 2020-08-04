import os
import sys
import time
import importlib
import gc
import mxnet as mx
from mxnet import gluon, nd, gpu, init, context
import mxnet.ndarray as FF
import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision
import random

from utils import *
from blackbox_attack import _perturbation_image

from gluoncv.model_zoo import get_model
from gluoncv.data.transforms import video
from gluoncv.data import SomethingSomethingV2
from gluoncv.utils import split_and_load
from gluoncv.data.dataloader import tsn_mp_batchify_fn

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic=True

# load configurations
args = load_args()
config = load_json_config(args.config)

# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module("{}".format(file_name))

# setup device - CPU or GPU
device, device_ids = setup_cuda_devices(args)
print(" > Using device: {}".format(device.type))
print(" > Active GPU ids: {}".format(device_ids))

best_loss = float('Inf')

class Logger(object):
    def __init__(self, filepath = './log.txt', mode = 'w', stdout = None):
        if stdout == None:
            self.terminal = sys.stdout
        else:
            self.terminal = stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        os.fsync(self.log)
    def flush(self):
        pass

def batch_fn(batch, ctx):
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    item_id = split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label, item_id

def main():
    global args, best_loss
    # create model, load existing models from gluoncv
    print(" > Creating model ... !")
    num_gpus = args.num_gpus
    batch_size = args.batch_size
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = args.num_workers
    print('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
    
    # =================== load model and parameters =======================
    classes = args.num_classes
    model_name = args.model
    model = get_model(name = model_name, 
                      nclass = classes, 
                      pretrained = args.use_pretrained, 
                      num_segments = args.num_segments)
    model.cast(args.dtype)
    model.collect_params().reset_ctx(context)
    
    if args.mode == 'hybrid':
        model.hybridize(static_alloc=True, static_shape=True)
    if args.resume_params is not '' and not args.use_pretrained:
        model.load_parameters(args.resume_params, ctx=context)
        print('Pre-trained model %s is successfully loaded.' % (args.resume_params))
    else:
        print('Pre-trained model is successfully loaded from the model zoo.')

    # ===================== load dataset =====================
    global transform_post
    
    transform_post = video.VideoGroupValTransform(size=args.input_size, mean = [0,0,0], std=[1,1,1])
    val_dataset = SomethingSomethingV2(setting=args.val_list, 
                                       root=args.data_dir, 
                                       train=False,
                                       new_width=args.new_width, 
                                       new_height=args.new_height,
                                       new_length=args.new_length,
                                       new_step=args.new_step,
                                       target_width=args.input_size, 
                                       target_height=args.input_size, 
                                       video_loader=args.video_loader, 
                                       use_decord=args.use_decord, 
                                       num_segments=args.num_segments, 
                                       transform=transform_post)

    val_loader = gluon.data.DataLoader(val_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       #num_workers=num_workers,
                                       prefetch=int(args.prefetch_ratio * num_workers), 
                                       batchify_fn=tsn_mp_batchify_fn, 
                                       last_batch='discard')
    
    print('Load %d test samples in %d iterations.' % (len(val_dataset), len(val_loader)))
    
    # ========================== attack ===========================
    if args.eval_only:
        validate(val_loader, model, context)
        print(" > Evaluation DONE !")
        return

def validate(val_loader, model, ctx):

    num_testing = 0
    success_testing = 0
    all_num_queries = 0.
    avg_query = 0.

    mpeg4_video_file = args.mpeg4_video_file
    frame_save_file = args.frame_save_file 

    #stdout_backup = sys.stdout
    #sys.stdout = Logger('./log.txt', 'w', stdout_backup)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            start = time.time()
            data, target, item_id = batch_fn(batch, ctx)
            X = data[0]

            target = torch.from_numpy(target[0].asnumpy()).long().to(device)
            XX = normalization(copy.copy(X.asnumpy()), args)
            output = model(XX.astype(args.dtype, copy=False))
            output = FF.mean(output, axis=0, keepdims=True)
            output = torch.from_numpy(output.asnumpy()).to(device)
            
            print('{}-th video'.format(i+1))
            if output.argmax(1) == target:
                num_testing += 1
                video_path = os.path.join(mpeg4_video_file, str(item_id[0].asnumpy()[0])+'.webm')
                save_path = os.path.join(frame_save_file, str(item_id[0].asnumpy()[0]))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                print('====={}-th video {}====='.format(i+1, video_path))
                pred_adv_label, num_query, success = _perturbation_image(model, 
                                                                        torch.from_numpy(X.asnumpy()), 
                                                                        target, 
                                                                        video_path, 
                                                                        save_path, 
                                                                        transform_post,
                                                                        args,
                                                                        config, 
                                                                        device)
                if success:
                    success_testing +=1
                    all_num_queries += num_query
                    avg_query = all_num_queries/success_testing
                    print('[T1]{:.3f}s for [{}]-th success sample\t'
                            'Attack number [{}]\t'
                            'Avg.Queries {:.2f}\t'
                            'Ori label {}\t'
                            'Pred label {}\t'
                            'Success rate {:.3f}\t'.format(time.time()-start, success_testing, 
                                                           num_testing, avg_query, 
                                                           target.item(), pred_adv_label,
                                                           success_testing/num_testing, 
                                                        ))

if __name__ == '__main__':
    main()
