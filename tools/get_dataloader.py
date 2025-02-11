from os.path import join
from .common import *
from torch.utils.data import DataLoader
from .visualize import group_images, save_img
from .extract_patches import get_data_train
from .datasetV2 import data_preprocess,create_patch_idx,TrainDatasetV2
from tqdm import tqdm

# ========================get dataloader==============================
def get_dataloaderV2(args, is_target):
    if not is_target:
        imgs_train, masks_train, fovs_train = data_preprocess(data_path_list = args.train_data_path_list)
        train_idx = create_patch_idx(fovs_train, args)
        # val_idx, train_idx = np.vsplit(patches_idx, (int(np.floor(args.val_N_patches)),))
        '''根据预处理得到的patch中心索引, 在 getitem 中从原图中提取patch, 从而返回patch'''
        train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, train_idx, mode="train", args=args,)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)

        # val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, val_idx, mode="val", args=args)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size,
        #                         shuffle=False, num_workers=cfg.NUM_WORKERS)
    else:
        imgs_train, masks_train, fovs_train = data_preprocess(data_path_list=args.train_trg_data_path_list)
        train_idx = create_patch_idx(fovs_train, args)
        # imgs_test, masks_test, fovs_test = data_preprocess(data_path_list=args.test_trg_data_path_list)
        # val_idx = create_patch_idx(fovs_test, args, is_target=is_target)

        train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, train_idx, mode="train", args=args,
                                   is_target=is_target)
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        # val_set = TrainDatasetV2(imgs_test, masks_test, fovs_test, val_idx, mode="val", args=args)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size,
        #                         shuffle=False, num_workers=cfg.NUM_WORKERS)

    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train,train_idx,mode="val",args=args)
        visual_loader = DataLoader(visual_set, batch_size=1,shuffle=True, num_workers=0)
        N_sample = 50
        visual_imgs = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample,1,args.train_patch_height, args.train_patch_width))

        for i, (img, mask) in tqdm(enumerate(visual_loader)):
            visual_imgs[i] = np.squeeze(img.numpy(),axis=0)
            visual_masks[i,0] = np.squeeze(mask.numpy(),axis=0)
            if i>=N_sample-1:
                break
        save_img(group_images((visual_imgs[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                "sample_input_imgs.png")
        save_img(group_images((visual_masks[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
                "sample_input_masks.png")
    return train_loader
