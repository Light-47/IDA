# import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy
import torch
from torchvision import transforms


def visualize_img(features):
    # 判断是否已经改为one-hot编码
    if torch.max(features) == 1  and torch.min(features) == 0:
        argmax_pred = features
    else:
        softmax_features = torch.softmax(features.detach(), dim=1)
        _, argmax_pred = torch.max(softmax_features, dim=0)   # 0-1 range
    # Visualize Pseudo labels
    pred = argmax_pred.detach().cpu().numpy()
    pred = np.where(pred > 0, 255, 0).astype(np.uint8)
    pred = transforms.ToPILImage()(pred)
    return pred

#group a set of img patches 
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg

# Prediction result splicing (original img, predicted probability, binary img, groundtruth)
def concat_result(ori_img,pred_res,gt):
    ori_img = data = np.transpose(ori_img,(1,2,0))
    pred_res = data = np.transpose(pred_res,(1,2,0))
    gt = data = np.transpose(gt,(1,2,0))

    binary = deepcopy(pred_res)
    binary[binary>=0.5]=1
    binary[binary<0.5]=0  

    if ori_img.shape[2]==3:
        pred_res = np.repeat((pred_res*255).astype(np.uint8),repeats=3,axis=2)
        binary = np.repeat((binary*255).astype(np.uint8),repeats=3,axis=2)
        gt = np.repeat((gt*255).astype(np.uint8),repeats=3,axis=2)
    total_img = np.concatenate((ori_img,pred_res,binary,gt),axis=1)
    return total_img, binary

#visualize image, save as PIL image
def save_img(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    img = Image.fromarray(data.astype(np.uint8))  #the image is between 0-1
    img.save(filename)
    return img