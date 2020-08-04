from coviar import get_num_frames, load
from utils import save_images, CWLoss, normalization, Cross_Entropy
import numpy as np
import torch
import copy
import pdb
import time
import torch.nn.functional as F
import mxnet.ndarray as FF
import random

GOP_SIZE = 12
def norm2(x):
    assert len(x.shape) == 4
    norm_vec = torch.sqrt(x.float().pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

def _pert_loss(logits, ori_label, target_label, delta_motion):
    cw_loss = CWLoss
    #cw_loss = Cross_Entropy
    loss, target, real, other, other_class, second_logit, second_class = cw_loss(logits, target_label)
    loss = loss.squeeze(0)
    return loss, target, real, other, other_class, second_logit, second_class

def _perturbation_image(model,  
                      original_image, 
                      ori_label, 
                      video_path,
                      save_path,
                      transform_post,
                      args,
                      config,
                      device):
    
    original_image = original_image.to(device) 

    total_frames = get_num_frames(video_path)
    original_image_ = original_image.clone() # torch.Size([1, 3, 72, 84, 84])
    num_frame, channel, height, width = original_image.shape
    dim = height * width * channel
    loop = 0
    inner_loop = 0
    success = False
    num_query = 0
    num_pframe = 0
    
    max_query = 60000
    exploration = 0.1 
    fd_eta = 0.1
    online_lr = 0.1
    flow_lr = 0.025
    target_label = (ori_label + 1)%args.num_classes
    '''
    while target_label == ori_label:
        target_label = torch.tensor([random.sample(range(174), 1)[0]]).cuda()
    '''
    motion_vector = list()

    prior = torch.zeros(num_frame, channel, height, width).to(device)
    delta = torch.zeros(num_frame, channel, height, width).to(device)
    est_grad = torch.zeros(num_frame, channel, height, width).to(device)
    adv_img = torch.zeros(3, num_frame, channel, height, width).to(device)
    iframe = torch.zeros(num_frame, height, width, channel).to(device)
    noise_frames = torch.zeros(num_frame, channel, height, width).to(device)
    
    index_visual = torch.zeros(num_frame, 2, height, width).to(device)
    index_motion = torch.zeros(num_frame, height, width, 2).to(device)
    
    while not (num_query > max_query):
        pred_adv_logit = list()
        start1 = time.time()

        end_index = total_frames // GOP_SIZE
        if loop%args.interval == 0: # can also try 8 for tsn2d
            #mv_index = int(torch.rand(1)*end_index)
            mv_index = inner_loop%end_index
            mv = load(video_path, mv_index, 11, 1, True)

            mv = mv - mv.min()
            mv = np.dstack((mv, np.zeros((mv.shape[:2] + (1,)))))
            mv = [mv.astype(np.uint8)]*num_frame
            inner_loop += 1
            motion_vector = transform_post(mv)
            motion_vector = np.stack(motion_vector, axis = 0)*255
            motion_vector = torch.from_numpy(motion_vector).permute(0, 2, 3, 1).float().to(device)
    
            motion_vector[:,:,:,0] = (2*motion_vector[:,:,:,0]-height+1.)/(height-1.)
            motion_vector[:,:,:,1] = (2*motion_vector[:,:,:,1]-width+1.)/(width-1.)


        noise_frames = torch.randn(1, 3, height, width).repeat(num_frame, 1, 1, 1).to(device)
        noise_frames = F.grid_sample(noise_frames, motion_vector[:,:,:,:2])

        exp_noise = exploration * noise_frames
        q1 = prior + exp_noise
        q2 = prior - exp_noise
        adv_img[0] = original_image + fd_eta*q1/norm2(q1)
        adv_img[1] = original_image + fd_eta*q2/norm2(q2)
        adv_img[2] = original_image
        for i in range(3):
            img_group = normalization(adv_img[i].clone().cpu().numpy(), args)
            tmp_result = model(img_group.astype('float32', copy=False))
            tmp_result = FF.mean(tmp_result, axis=0, keepdims=True)
            tmp_result = torch.from_numpy(tmp_result.asnumpy()).to(device)
            pred_adv_logit.append(tmp_result)
    
        l1, _, _, _, _, _, _ = _pert_loss(pred_adv_logit[0], ori_label, target_label, delta)
        l2, _, _, _, _, _, _ = _pert_loss(pred_adv_logit[1], ori_label, target_label, delta)
        loss, target, real, other, other_class, second_logit, second_class = _pert_loss(pred_adv_logit[2], ori_label, target_label, delta)
        
        num_query += 3
        est_deriv = (l1-l2)/(fd_eta*exploration*exploration)
        est_grad = est_deriv.item() * exp_noise
        prior += online_lr * est_grad
        
        original_image = original_image - flow_lr*prior.sign()
        delta = original_image_ - original_image
        tmp_norm = norm2(delta)
        original_image = torch.max(torch.min(original_image, original_image_ + 0.03), original_image_ - 0.03)
        original_image = torch.clamp(original_image, 0, 1)
        
        pred_adv_label = pred_adv_logit[2].argmax()
        if (loop % 1000 ==0) or (loop == max_query) or pred_adv_label != ori_label:
        #if (loop % 1000 ==0) or (loop == max_query) or pred_adv_label == target_label:
            print('[T2]{:.3f}s for [{}]-th loop\t'
                  'Queries {:03d}\t'
                  'Overall loss {:.3f}\t'
                  'est_deriv {:.3f}\t'
                  'Target {}\t'
                  'Target logit {:.3f}\t'
                  'ori logit {:.3f}\t'
                  'ori class {}\t'
                  'second logit {:.3f}\t'
                  'second class {}\t'.format(time.time() - start1, loop,
                                            num_query, loss, est_deriv.item(), target, 
                                            real, other, other_class, second_logit, second_class))  
        
        loop += 1
        if pred_adv_label != ori_label:
        #if pred_adv_label == target_label:
            #print('Predicted label is {}\t'.format(pred_adv_label))
            diff = adv_img[2] - original_image_
            print('diff max {:.3f}, diff min {:.3f}'.format(diff.max(), diff.min()))
            success = True
            #save_images(num_frame, original_image_.cpu().permute(0,2,3,1).numpy(), adv_img[2].cpu().permute(0,2,3,1).numpy(), save_path)
            break

        if num_query >= max_query:
            #save_images(num_frame, original_image_.cpu().permute(0,2,3,1).numpy(), adv_img[2].cpu().permute(0,2,3,1).numpy(), save_path)
            break
    return pred_adv_label, num_query, success 

