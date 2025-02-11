import os
from os.path import join


def get_path_list(root_path, img_path, label_path, fov_path):
    tmp_list = [img_path, label_path, fov_path]
    res = []
    for i in range(len(tmp_list)):
        data_path = join(data_root_path, tmp_list[i])
        filename_list = os.listdir(data_path)
        filename_list.sort()
        res.append([join(data_path, j) for j in filename_list])
    return res


def write_path_list(name_list, save_path, file_name):
    f = open(join(save_path, file_name), 'w')
    for i in range(len(name_list[0])):
        f.write(str(name_list[0][i]) + " " + str(name_list[1][i]) + " " + str(name_list[2][i]) + '\n')
    f.close()


if __name__ == "__main__":
    # ------------Path of the dataset --------------------------------
    data_root_path = 'D:/repo/IDDA/data'
    # if not os.path.exists(data_root_path): raise ValueError("data path is not exist, Please make sure your data path is correct")
    # train
    img = "CAM/train/images"
    gt = "CAM/train/fake_label"
    fov = "CAM/train/mask"
    # ---------------save path-----------------------------------------
    save_path = "D:/repo/IDDA/prepare_dataset/data_path_list/CAM"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # -----------------------------------------------------------------
    data_list = get_path_list(data_root_path, img, gt, fov)
    print('Number of train imgs:', len(data_list[0]))
    write_path_list(data_list, save_path, 'train.txt')
    # test
    img = "CAM/test/images"
    gt = "CAM/test/manual"
    fov = "CAM/test/mask"
    data_list = get_path_list(data_root_path, img, gt, fov)
    print('Number of test imgs:', len(data_list[0]))
    write_path_list(data_list, save_path, 'test.txt')

    print("Finish!")

