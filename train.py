import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__)) # 获取当前绝对路径C
sys.path.append(curPath)
rootPath = os.path.split(curPath)[0]				 # 上一级目录B
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import yaml
import torch
from parser_train import get_arguments
from model.UNet_Zoo.WUNet import wnet, my_wnet
from domain_adaptation.config_vessel import cfg, cfg_from_file
from domain_adaptation.train_UDA_vessel import train_domain_adaptation
from tools.get_dataloader import get_dataloaderV2
from tools.get_whole_loaders import get_train_loaders

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_model(cfg, model_name, multi_level=True, ema=False):
    if model_name == 'my_wnet':
        model = my_wnet(in_c=1, n_classes=cfg.NUM_CLASSES)
    # add more models here
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def main():
    #LOAD ARGS
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser = get_arguments(parser)
    args   = parser.parse_args()

    print('Called with args')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)

    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT,args.checkpoint)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    #tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR,exist_ok=True)

        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE  = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg)

    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    if os.environ.get('ADVENT_DRY_RUN','0') == '1' :
        return

    print('Loading model:', cfg.TRAIN.MODEL, '...')
    if cfg.TRAIN.MODEL == 'my_wnet':
        '''note: input channel=1'''
        model = get_model(cfg, cfg.TRAIN.MODEL)
        ema_model = get_model(cfg, cfg.TRAIN.MODEL, ema=True)

        if cfg.TRAIN.RESTORE_FROM:
            saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM, map_location=torch.device('cpu'))
            model.load_state_dict(saved_state_dict)
            if not args.source_pretrain:
                ema_model.load_state_dict(saved_state_dict)       
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')

    '''whole_loader'''
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    whole_src_loader = get_train_loaders(csv_path_train=args.csv_train, is_target=False, batch_size=args.batch_size, 
                                                                 tg_size=tg_size, label_values=[0, 255], num_workers=cfg.NUM_WORKERS)
    whole_trg_loader = get_train_loaders(csv_path_train=args.csv_target_train, is_target=True, batch_size=args.batch_size, 
                                                                   tg_size=tg_size, label_values=[0, 255], num_workers=cfg.NUM_WORKERS)
    print('Whole image dataloader finished')
    '''patch_loader'''
    patch_src_loader = get_dataloaderV2(args, is_target=False)
    patch_trg_loader = get_dataloaderV2(args, is_target=True)
    print('Patch image dataloader finished')

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)
    
    # UDA TRAINING
    train_domain_adaptation(model, ema_model, whole_src_loader, whole_trg_loader, patch_src_loader, patch_trg_loader, cfg, args)

if __name__ == '__main__':
     main()
