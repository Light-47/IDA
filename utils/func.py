import numpy as np
import torch
import torch.nn as nn
from utils.loss_distri import cross_entropy_2d
# import cv2
import torch.nn.functional as F
import torch.sparse as sparse

def dice_eval(pred,label,n_class):
    '''
    pred:  b*c*h*w
    label: b*h*w
    '''
    pred     = torch.argmax(pred,dim=1)  # b*h*w
    dice     = 0
    dice_arr = []
    each_class_number = []
    eps      = 1e-7
    for i in range(n_class):
        A = (pred  == i)
        B = (label == i)
        each_class_number.append(torch.sum(B).cpu().data.numpy())
        inse  = torch.sum(A*B).float()
        union = (torch.sum(A)+torch.sum(B)).float()
        dice  += 2*inse/(union+eps)
        dice_arr.append((2*inse/(union+eps)).cpu().data.numpy())

    return dice,dice_arr,np.hstack(each_class_number)

def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

def loss_calc(pred,label,device):

    '''
    This function returns cross entropy loss for semantic segmentation
    '''
    # pred shape is batch * c * h * w
    # label shape is b*h*w
    label = label.long().cuda(device)
    return cross_entropy_2d(pred, label,device)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)

def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)

def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-7)) / np.log2(c)

def sel_prob_2_entropy(prob):
    n, c, h, w = prob.size()
    weighted_self_info = -torch.mul(prob, torch.log2(prob + 1e-30)) / c
    entropy            = torch.sum(weighted_self_info,dim=1) #N*C*H*W
    # mean_entropy       = torch.sum(entropy,dim=[1,2])
    return entropy



def mpcl_loss_calc(device, feas,labels,class_center_feas,loss_func, pixel_mask=None, tag='source'):

    '''
    feas:  batch*c*h*w
    label: batch*img_h*img_w
    class_center_feas: n_class*n_feas
    '''
    n,c,fea_h,fea_w = feas.size()
    if tag == 'source':
        labels      = labels.float()
        labels      = F.interpolate(labels, size=fea_w, mode='nearest')
        labels      = labels.permute(0,2,1).contiguous()
        labels      = F.interpolate(labels, size=fea_h, mode='nearest')
        labels      = labels.permute(0, 2, 1).contiguous()         # batch*fea_h*fea_w
    labels  = labels.cuda(device)
    labels  = labels.view(-1).long()

    feas = torch.nn.functional.normalize(feas,p=2,dim=1)   # l2
    feas = feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    feas = torch.reshape(feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c
    feas = feas.unsqueeze(1) # [batch*h*w] 1 * c

    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1)
    class_center_feas = torch.transpose(class_center_feas, 0, 1)  # n_fea*n_class

    loss =  loss_func(feas, labels, class_center_feas, pixel_mask= pixel_mask)
    return loss

def dist_loss_calc(device, feat, mask, mean, covariance):

    '''
    feat:  batch*c*h*w -> N*D
    mask: N*1
    '''
    mask = torch.tensor(mask, dtype=torch.int64).cuda(device)
    # caculate covariance using imgs features
    # source
    n,c,fea_h,fea_w = feat.size()
    feat = torch.nn.functional.normalize(feat,p=2,dim=1)
    feat = feat.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    feat = torch.reshape(feat,[n*fea_h*fea_w,c]) # [batch*h*w] * c

    ratio = 1.0
    contrast_temp = 1.0   # 100.0
    # caculate the first term
    # [batch * h * w] * 1
    temp1 = feat.mm(mean.permute(1, 0).contiguous())
    # feat (N, A)^2 x CoVariance (A, C)
    covariance = covariance * ratio / contrast_temp
    temp2 = 0.5 * feat.pow(2).mm(covariance.permute(1, 0).contiguous())

    logits = temp1 + temp2
    logits = logits / contrast_temp

    # The wrapper function for :func:`F.cross_entropy`
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    ce_loss = F.cross_entropy(
        logits.cuda(device),
        mask,
        weight=None,
        reduction='none',
        ignore_index=255)

    jcl_loss = 0.5 * torch.sum(feat.pow(2).mul(covariance[mask]), dim=1) / contrast_temp
    loss_dist = ce_loss + jcl_loss
    loss_dist = torch.mean(loss_dist)
    return loss_dist

