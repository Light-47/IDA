
def get_arguments(parser):
    """
    Parse input arguments
    """
    parser.add_argument('--cfg', type=str, default='./scripts/configs/DRIVE2CAM.yml',
                        help='optional config file')
    '''pre-train settings'''
    parser.add_argument("--source_pretrain", default=False, type=bool,
                        help="Whether to pretrain the source model.")
    parser.add_argument("--self_cutmix", default=False, type=bool,
                        help="Whether to use self-cutmix in pre-training.")
    parser.add_argument("--self_CL", default=False, type=bool,
                        help="Whether to use self-contrastive learning in pre-training.")
        
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--checkpoint", default='DRIVE2CAM_WNET',
                        type=str, help="path to pth")
    '''datalader settings'''
    # whole images path
    parser.add_argument('--csv_train', type=str, default='data/DRIVE/train.csv', help='path to training data csv')
    parser.add_argument('--csv_target_train', type=str, default='data/CAM/train.csv', help='path to training data csv')
    # patch images path
    parser.add_argument('--train_data_path_list',
                        default='./prepare_dataset/data_path_list/DRIVE/train.txt')
    parser.add_argument('--train_trg_data_path_list',
                        default='./prepare_dataset/data_path_list/CAM/train.txt')
    parser.add_argument('--test_trg_data_path_list',
                        default='./prepare_dataset/data_path_list/CAM/test.txt')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--im_size', help='Specify the size of the whole image', type=str, default='384')
    parser.add_argument('--concat_size', default=128, type=int,help='concatenation size')
    parser.add_argument('--train_patch_height', default=384, help='consistent with im_size')
    parser.add_argument('--train_patch_width', default=384)
    parser.add_argument('--N_patches', default=5000,   
                        help='Number of training image patches, setting N_patches == max_iters')
    parser.add_argument('--val_N_patches', default=200, 
                        help='Number of val image patches, not used')
    parser.add_argument('--inside_FOV', default='center',
                        help='Choose from [not,center,all]')
    parser.add_argument('--sample_visualization', default=False,
                        help='Visualization of training samples')
    return parser