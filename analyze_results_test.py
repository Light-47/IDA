import argparse
from PIL import Image
import os, sys
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
# for clDice
from skimage.morphology import skeletonize, skeletonize_3d
# for Betti
import torch
from utils.BM.betti_error import BettiNumberMetric

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.metrics import matthews_corrcoef
from utils.evaluation import dice_score
# from utils.model_saving_loading import str2bool

# future-self: dice and f1 are the same thing, but if you use f1_score from sklearn it will be much slower, the reason
# being that dice here expects bools and it won't work in multi-class scenarios. Same goes for accuracy_score.
# (see https://brenocon.com/blog/2012/04/f-scores-dice-and-jaccard-set-similarity/)

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, default='CAMDB', help='which dataset to test')
parser.add_argument('--path_test_preds', type=str,
                    default='',
                    help='path to test predictions') 
parser.add_argument('--cut_off', type=str, default='dice', help='threshold maximizing x, x=dice/acc/youden')
parser.add_argument('--opt_threshold', type=float, default=None, help='if > threshold, vessel, otherwise background')
parser.add_argument("--GPU", type=str, default="cuda:2", help="path to pth")

def get_labels_preds(path_to_preds, csv_path):
    df = pd.read_csv(csv_path)
    im_list, mask_list, gt_list = df.im_paths, df.mask_paths, df.gt_paths
    # im_list, gt_list = df.im_paths, df.gt_paths
    all_bin_preds = []
    all_preds = []
    all_gts = []
    twoDgt, twoDpred = [], []
    for i in range(len(gt_list)):
        im_path = im_list[i].rsplit('/', 1)[-1]
        pred_path = osp.join(path_to_preds, im_path[:-4] + '.png')
        gt_path = gt_list[i]
        mask_path = mask_list[i]

        gt = np.array(Image.open(gt_path).convert('L')).astype(bool)
        mask = np.array(Image.open(mask_path).convert('L')).astype(bool)
        from skimage import img_as_float
        try: pred = img_as_float(np.array(Image.open(pred_path)))
        except FileNotFoundError:
            sys.exit('---- no predictions found at {} (maybe run first generate_results.py?) ---- '.format(path_to_preds))
        gt_flat = gt.ravel()
        mask_flat = mask.ravel()
        pred_flat = pred.ravel()
        # do not consider pixels out of the FOV
        noFOV_gt = gt_flat[mask_flat == True]
        noFOV_pred = pred_flat[mask_flat == True]

        # accumulate gt pixels and prediction pixels
        all_preds.append(noFOV_pred)
        all_gts.append(noFOV_gt)
        # all_gts.append(gt_flat)
        twoDgt.append(gt)
        twoDpred.append(pred)

    return np.hstack(all_preds), np.hstack(all_gts), np.stack(twoDpred, axis=0), np.stack(twoDgt, axis=0)

def cutoff_youden(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def cutoff_dice(preds, gts):
    dice_scores = []
    thresholds = np.linspace(0, 1, 256)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds>thresh
        dice_scores.append(dice_score(gts, hard_preds))
    dices = np.array(dice_scores)
    optimal_threshold = thresholds[dices.argmax()]
    return optimal_threshold

def cutoff_accuracy(preds, gts):
    accuracy_scores = []
    thresholds = np.linspace(0, 1, 256)
    for i in tqdm(range(len(thresholds))):
        thresh = thresholds[i]
        hard_preds = preds > thresh
        accuracy_scores.append(accuracy_score(gts.astype(np.bool), hard_preds.astype(np.bool)))
    accuracies = np.array(accuracy_scores)
    optimal_threshold = thresholds[accuracies.argmax()]
    return optimal_threshold

def compute_performance(preds, gts, save_path=None, opt_threshold=None, cut_off='dice', mode='train'):

    fpr, tpr, thresholds = roc_curve(gts, preds)
    global_auc = auc(fpr, tpr)

    if save_path is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label='ROC curve')
        ll = 'AUC = {:4f}'.format(global_auc)
        plt.legend([ll], loc='lower right')
        fig.tight_layout()

        plt.savefig(osp.join(save_path, 'ROC_test.png'))

    if opt_threshold is None:
        if cut_off == 'acc':
            # this would be to get accuracy-maximizing threshold
            opt_threshold = cutoff_accuracy(preds, gts)
        elif cut_off == 'dice':
            # this would be to get dice-maximizing threshold
            opt_threshold = cutoff_dice(preds, gts)
            print('#####')
            print('dice maximizing threshold is {:.4f}'.format(opt_threshold))
            print('#####')
        else:
            opt_threshold = cutoff_youden(fpr, tpr, thresholds)

    bin_preds = preds > opt_threshold

    acc = accuracy_score(gts, bin_preds)

    dice = dice_score(gts, bin_preds)
    
    mcc = matthews_corrcoef(gts.astype(int), bin_preds.astype(int))

    tn, fp, fn, tp = confusion_matrix(gts, preds > opt_threshold).ravel()
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return global_auc, acc, dice, mcc, specificity, sensitivity, precision, opt_threshold

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

if __name__ == '__main__':
    # gather parser parameters
    args = parser.parse_args()

    test_dataset = args.test_dataset
    path_test_preds = args.path_test_preds
    cut_off = args.cut_off

    '''Analyzing test sets'''
    print('*Analyzing performance in {} test set'.format(test_dataset))
    print('* Reading predictions from {}'.format(path_test_preds))
    save_path = osp.join(path_test_preds, 'perf')
    os.makedirs(save_path, exist_ok=True)
    perf_csv_path = osp.join(save_path, 'test_performance.csv')

    csv_name = 'test.csv'

    path_test_csv = osp.join('data', test_dataset, csv_name)

    preds, gts, twoDpred, twoDgt = get_labels_preds(path_test_preds, csv_path = path_test_csv)
    ''' 设置阈值 '''
    global_auc_test, acc_test, dice_test, mcc_test, spec_test, sens_test, pre_test, opt_threshold = \
        compute_performance(preds, gts, save_path=save_path, opt_threshold=args.opt_threshold)
    
    ### 计算 clDice
    cldice_list = []
    for i in range(twoDpred.shape[0]):
        pred, gt = twoDpred[i], twoDgt[i]
        gt = (gt > 0).astype(np.uint8)
        pred = (pred >= opt_threshold).astype(np.uint8)
        cdice = clDice(pred, gt)
        cldice_list.append(cdice)
    cldice = sum(cldice_list) / len(cldice_list)
    ### 计算 BettiMatching error
    device = args.GPU
    betti_number_metric = BettiNumberMetric(
        num_processes=16,
        ignore_background=True,
        eight_connectivity=False
    )
    twoDpred, twoDgt = torch.from_numpy(twoDpred).to(device), torch.from_numpy(twoDgt).to(device)    
    twoDpred = (twoDpred >= opt_threshold).long()
    one_hot_pred = torch.zeros((twoDpred.size(0), 2, twoDpred.size(1), twoDpred.size(2)), 
                        dtype=torch.long, device=twoDpred.device)
    one_hot_pred.scatter_(1, twoDpred.unsqueeze(1), 1)
    twoDgt = twoDgt.long()
    onehot_gt = torch.zeros((twoDgt.size(0), 2, twoDgt.size(1), twoDgt.size(2)), 
                            dtype=torch.long, device=twoDgt.device)
    onehot_gt.scatter_(1, twoDgt.unsqueeze(1), 1)
    betti_number_metric(y_pred=one_hot_pred, y=onehot_gt)
    b0, b1, bm, norm_bm = betti_number_metric.aggregate()
    
    # log
    perf_df_test = pd.DataFrame({'opt_threshold': opt_threshold,
                                 'auc': global_auc_test,
                                 'acc': acc_test,
                                 'p':pre_test,
                                 'sens': sens_test,
                                 'spec': spec_test,
                                 'dice/F1': dice_test,
                                 'clDice': cldice,
                                 'Beta Error': bm,
                                 'Beta0 Error': b0,
                                 'Beta1 Error': b1,
                                 'MCC': mcc_test
                                 }, index=[0])
    perf_df_test.to_csv(perf_csv_path, index=False)
    print('* Done')
    print('AUC in Test set is {:.4f}'.format(global_auc_test))
    print('Accuracy in Test set is {:.4f}'.format(acc_test))
    print('Precision in Test set is {:.4f}'.format(pre_test))    
    print('Sensitivity in Test set is {:.4f}'.format(sens_test))    
    print('Specificity in Test set is {:.4f}'.format(spec_test))
    print('Dice/F1 score in Test set is {:.4f}'.format(dice_test))
    print('clDice in Test set is {:.4f}'.format(cldice))
    print('Betti Matching Error in Test set is {:.5f}'.format(bm.numpy().item()))
    print('Betti0 Error in Test set is {:.4f}'.format(b0.numpy().item()))
    print('Betti1 Error in Test set is {:.4f}'.format(b1.numpy().item()))
    print('opt_threshold is {:.4f}'.format(opt_threshold))
    print('ROC curve plots saved to ', save_path)
    print('Perf csv saved at ', perf_csv_path)
