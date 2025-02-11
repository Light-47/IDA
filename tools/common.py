import numpy as np
import os,joblib
import torch,random
import torch.nn as nn
import cv2,PIL
# from libtiff import TIFFfile

def readImg(img_path):
    """
    When reading local image data, because the format of the data set is not uniform,
    the reading method needs to be considered. 
    Default using pillow to read the desired RGB format img
    """
    img_format = img_path.split(".")[-1]
    try:
        #在win下读取tif格式图像在转np的时候异常终止，暂时没找到合适的读取方式，Linux下直接用PIl读取无问题
        img = PIL.Image.open(img_path) 
    except Exception as e:
        ValueError("Reading failed, please check path of dataset,",img_path)
    return img

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print(self.val)

# formulate a learning rate decay strategy
def make_lr_schedule(lr_epoch,lr_value):
    lr_schedule = np.zeros(lr_epoch[-1])
    for l in range(len(lr_epoch)):
        if l == 0:
            lr_schedule[0:lr_epoch[l]] = lr_value[l]
        else:
            lr_schedule[lr_epoch[l - 1]:lr_epoch[l]] = lr_value[l]
    return lr_schedule

# Save configuration information
def save_args(args,save_path):
    if not os.path.exists(save_path):
        os.makedirs('%s' % save_path)

    print('Config info -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    with open('%s/args.txt' % save_path, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)
    joblib.dump(args, '%s/args.pkl' % save_path)
    print('\033[0;33m================config infomation has been saved=================\033[0m')

# Seed for repeatability
def setpu_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    random.seed(seed)

# Round off
def dict_round(dic,num):
    for key,value in dic.items():
        dic[key] = round(value,num)
    return dic
