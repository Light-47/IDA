import os
import os.path as osp
import pandas as pd


print('preparing data')

#######################################################################################################################
# process drive data, generate CSVs
path_ims = 'data/CAM/train/images'
path_masks = 'data/CAM/train/mask'
path_gts = 'data/CAM/train/fake_label'

path_test_ims = 'data/CAM/test/images'
path_test_masks = 'data/CAM/test/mask'
path_test_gts = 'data/CAM/test/manual'

train_im_names = sorted(os.listdir(path_ims))
train_mask_names = sorted(os.listdir(path_masks))
train_gt_names = sorted(os.listdir(path_gts))

test_im_names = sorted(os.listdir(path_test_ims))
test_mask_names = sorted(os.listdir(path_test_masks))
test_gt_names = sorted(os.listdir(path_test_gts))

# append paths
train_im_names = [osp.join(path_ims, n) for n in train_im_names]
train_mask_names = [osp.join(path_masks, n) for n in train_mask_names]
train_gt_names = [osp.join(path_gts, n) for n in train_gt_names]

test_im_names = [osp.join(path_test_ims, n) for n in test_im_names]
test_mask_names = [osp.join(path_test_masks, n) for n in test_mask_names]
test_gt_names = [osp.join(path_test_gts, n) for n in test_gt_names]

train_im_names = train_im_names[:]
train_mask_names = train_mask_names[:]
train_gt_names = train_gt_names[:]

test_im_names  = test_im_names[:] 
test_mask_names = test_mask_names[:]
test_gt_names = test_gt_names[:]


df_drive_train = pd.DataFrame({'im_paths': train_im_names,
                               'gt_paths': train_gt_names,
                               'mask_paths': train_mask_names})

df_drive_test = pd.DataFrame({'im_paths': test_im_names,
                             'gt_paths': test_gt_names,
                             'mask_paths': test_mask_names})

df_drive_train.to_csv('data/CAM/train.csv', index=False)
df_drive_test.to_csv('data/CAM/val.csv', index=False)
df_drive_test.to_csv('data/CAM/test.csv', index=False)
# df_drive_test.to_csv('data/STARE/test_all.csv', index=False)

print('done!')