import sys
sys.path.append('..')
import os.path as osp
import numpy as np
from easydict import EasyDict
from utils import project_root
from utils.serialization import yaml_load


cfg = EasyDict()

# COMMON CONFIGS
# source domain
cfg.SOURCE = 'DRIVE'  # DRIVE
# target domain
cfg.TARGET = 'CAM'
# Number of workers for dataloading
cfg.NUM_WORKERS = 4
# Number of object classes
cfg.NUM_CLASSES = 2
# Exp dirs
cfg.EXP_NAME          = ''
cfg.EXP_ROOT          =  './scripts/experiments'
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, '11_15')
cfg.EXP_ROOT_LOGS     = osp.join(cfg.EXP_ROOT, 'logs')
cfg.FULLY_SUPERVISED  = False

# CUDA
cfg.GPU_ID = 1
cfg.UDA    = True
# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE        = 4   # 4
cfg.TRAIN.INPUT_SIZE_SOURCE = (384,384)
cfg.TRAIN.INPUT_SIZE_TARGET = (384,384)
cfg.TRAIN.SRC_MARGIN        = 0.0
cfg.TRAIN.TRG_MARGIN        = 0.0
cfg.TRAIN.SRC_TEMP          = 1.0
cfg.TRAIN.SRC_BASE_TEMP     = 1.0
cfg.TRAIN.TRG_TEMP          = 1.0
cfg.TRAIN.BASE_TEMP         = 1.0
cfg.TRAIN.TRG_BASE_TEMP     = 1.0
cfg.TRAIN.LAMBDA_MPCL_SRC   = 0.1
cfg.TRAIN.LAMBDA_MPCL_TRG   = 0.1
cfg.TRAIN.RESTORE_FROM      = '' 
cfg.TRAIN.UPDATE_CENTER_METHOD = 'iter'

# Class info
# Segmentation network params
cfg.TRAIN.MODEL           = 'my_wnet'  # 'DeepLabv2' & 'my_wnet'
cfg.TRAIN.MULTI_LEVEL     = True
cfg.TRAIN.PreTrain        = True
cfg.TRAIN.LR_Decay        = True
cfg.TRAIN.CLASS_CENTER_FEA_INIT = ''
cfg.TRAIN.CLASS_CENTER_M   = 0.7

cfg.TRAIN.RESTORE_FROM    =  ''  
cfg.TRAIN.IMG_MEAN           = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE          = 2.5e-4  
cfg.TRAIN.MOMENTUM               = 0.9
cfg.TRAIN.WEIGHT_DECAY           = 0.0005
cfg.TRAIN.POWER                  = 0.9

cfg.TRAIN.LAMBDA_Patch_MAIN= 0.5
cfg.TRAIN.LAMBDA_Patch_AUX = 0.5

cfg.TRAIN.LAMBDA_SEG_S_MAIN    = 1.0
cfg.TRAIN.LAMBDA_SEG_S_AUX     = 0.1  # weight of conv4 prediction. Used in multi-level setting.
cfg.TRAIN.LAMBDA_DICE_S_MAIN   = 1.0
cfg.TRAIN.LAMBDA_DICE_S_AUX    = 0.1

# Domain adaptation
cfg.TRAIN.DA_METHOD = 'MPSCL'
# Other params
cfg.TRAIN.MAX_ITERS           = 5000 
cfg.TRAIN.EARLY_STOP          = 3000  
cfg.TRAIN.SAVE_PRED_EVERY     = 100 
cfg.TRAIN.SNAPSHOT_DIR        = ''
cfg.TRAIN.RANDOM_SEED         = 2024
cfg.TRAIN.TENSORBOARD_LOGDIR  = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 100

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'best'  # {'single', 'best'}
# model
cfg.TEST.MODEL = ('DeepLabv2',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 1000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 120000  # used in 'best' mode
# Test sets
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (128, 128)
cfg.TEST.OUTPUT_SIZE_TARGET = (128, 128)
cfg.TEST.INFO_TARGET = str(project_root / 'advent/dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
