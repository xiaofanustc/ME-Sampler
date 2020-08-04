import os
import sys
import json
import pickle
import argparse
import mxnet as mx
from mxnet import nd
import torch
import shutil
import glob
import numpy as np
import pdb
import torch.nn.functional as F
from scipy.misc import imsave

import torch.nn as nn
import torchvision
from gluoncv.data.transforms import video

transform_post = video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def normalization(imgs, args):
    batch_size = imgs.shape[0]
    imgs_ = transform_post(imgs)
    data = np.stack(imgs_, axis=0)
    if args.model_type == 'i3d':
        data = data.reshape((-1,) + (args.new_length, 3, args.input_size, args.input_size))
        data = np.transpose(data, (0, 2, 1, 3, 4))

    return  nd.array(data, ctx=mx.gpu())

def save_images(num_frame, original_image, adv_img2, save_path):
    for i in range(num_frame):
        imsave(os.path.join(save_path,'orig_{:05d}.png'.format(i)),original_image[i])
        imsave(os.path.join(save_path,'adv_{:05d}.png'.format(i)),adv_img2[i])

def save_images_noise(num_frame, noise_img, save_path):
    for i in range(num_frame):
        np.save(os.path.join(save_path,'noise_{:05d}.npy'.format(i)), noise_img[i])


def CWLoss(logits, target, kappa=0):
    logits = F.softmax(logits, dim=1)
    
    target_onehot = torch.zeros(1, 174).cuda()
    target_onehot[0, target] = 1
    real = (target_onehot * logits).sum(1)[0]
    tmp_logit = ((1. - target_onehot) * logits - target_onehot*10000.)
    
    other, other_class = logits.max(1)
    sort_prob, sort_class = logits.sort()
    second_logit = sort_prob[0][-2].unsqueeze(0)
    second_class = sort_class[0][-2].unsqueeze(0)
    
    return torch.clamp(torch.sum(logits)-second_logit, kappa), target.item(), real.item(), other.item(), other_class.item(), second_logit.item(), second_class.item()
    #return torch.clamp(other-5*real, kappa), target.item(), real.item(), other.item(), other_class.item(), second_logit.item(), second_class.item()

def Cross_Entropy(logits, target, kappa=0):
    criterion = nn.CrossEntropyLoss()
    other, other_class = logits.max(1)
    loss = criterion(logits, other_class.long())
    #pdb.set_trace()
    return loss, loss.item(), loss.item(), loss.item(), loss.item(), loss.item(), loss.item()

def load_args():
    parser = argparse.ArgumentParser(description='Smth-Smth example attacking')
    parser.add_argument('--config', '-c', help='json config file path')
    parser.add_argument('--eval_only', '-e', action='store_true', 
                        help="evaluate trained model on validation data.")
    parser.add_argument('--gpus', '-g', help="GPU ids to use. Please"
                         " enter a comma separated list")
    parser.add_argument('--use_cuda', action='store_true',
                        help="to use GPUs")

    parser.add_argument('--num-classes', type=int, default=101,
                        help='number of classes.')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--model_type', type=str, default='i3d', 
                        help='i3d or tsn2d model for different normalization')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--num-segments', type=int, default=1,
                        help='number of segments to evenly split the video.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--val-list', type=str, 
                        default='~/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_rgb_split1.txt', 
                        help='the list of validation data')
    parser.add_argument('--new-height', type=int, default=256,
                        help='new height of the resize image. default is 256')
    parser.add_argument('--new-width', type=int, default=340,
                        help='new width of the resize image. default is 340')
    parser.add_argument('--new-length', type=int, default=1,
                        help='new length of video sequence. default is 1')
    parser.add_argument('--new-step', type=int, default=1,
                        help='new step to skip video sequence. default is 1')
    parser.add_argument('--prefetch-ratio', type=float, default=2.0,
                        help='set number of workers to prefetch data batch, default is 2 in MXNet.')
    parser.add_argument('--video-loader', action='store_true',
                        help='if set to True, read videos directly instead of reading frames.')
    parser.add_argument('--use-decord', action='store_true',
                        help='if set to True, use Decord video loader to load data. Otherwise use mmcv video loader.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/ucf101/rawframes',
                        help='training (and validation) pictures to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')

    parser.add_argument('--mpeg4_video_file', type=str, default='/data/uts700/hu/smth-smth/smth-smth-val-mpeg4/',
                        help='path to load motion vector for attacking')
    parser.add_argument('--frame_save_file', type=str, default='/data/uts700/hu/smth-smth/save_smth_imgs_i3d',
                        help='path to save adversarial video frames')
    parser.add_argument('--interval', type=int, default=10, 
                        help='the interval to change motion vector')
    args = parser.parse_args()
    return args


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config

def config_init(config):
    """ Some of the variables that should exist and contain default values """
    if "augmentation_mappings_json" not in config:
        config["augmentation_mappings_json"] = None
    if "augmentation_types_todo" not in config:
        config["augmentation_types_todo"] = None
    return config

def setup_cuda_devices(args):
    device_ids = []
    device = torch.device("cuda" if args.use_cuda else "cpu")
    if device.type == "cuda":
        device_ids = [int(i) for i in args.gpus.split(',')]
    return device, device_ids

