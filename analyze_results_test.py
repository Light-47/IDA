import argparse
from PIL import Image
import os, sys
import os.path as osp
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.metrics import matthews_corrcoef
from utils.evaluation import dice_score

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, default='CHASEDB', help='which dataset to test')
parser.add_argument('--path_test_preds', type=str,
                    default='results/CHASEDB/', 
                    help='path to test predictions') 
parser.add_argument('--cut_off', type=str, default='dice', help='threshold maximizing x, x=dice/acc/youden')
parser.add_argument('--opt_threshold', type=float, default=None, help='if > threshold, vessel, otherwise background')

def get_labels_preds(path_to_preds, csv_path):
    df = pd.read_csv(csv_path)
    im_list, mask_list, gt_list = df.im_paths, df.mask_paths, df.gt_paths
    all_preds = []
    all_gts = []
    for i in range(len(gt_list)):
        im_path = im_list[i].rsplit('/', 1)[-1]
        pred_path = osp.join(path_to_preds, im_path[:-4] + '.png')
        gt_path = gt_list[i]
        mask_path = mask_list[i]

        gt = np.array(Image.open(gt_path)).astype(bool)
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

    return np.hstack(all_preds), np.hstack(all_gts)

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
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return global_auc, acc, dice, mcc, specificity, sensitivity, opt_threshold

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

    preds, gts = get_labels_preds(path_test_preds, csv_path = path_test_csv)

    global_auc_test, acc_test, dice_test, mcc_test, spec_test, sens_test, _ = \
        compute_performance(preds, gts, save_path=save_path, opt_threshold=args.opt_threshold)
    perf_df_test = pd.DataFrame({'opt_threshold': args.opt_threshold,
                                 'auc': global_auc_test,
                                 'acc': acc_test,
                                 'sens': sens_test,
                                 'spec': spec_test,
                                 'dice/F1': dice_test,
                                 'MCC': mcc_test}, index=[0])
    perf_df_test.to_csv(perf_csv_path, index=False)
    print('* Done')
    print('AUC in Test set is {:.4f}'.format(global_auc_test))
    print('Accuracy in Test set is {:.4f}'.format(acc_test))
    print('Sensitivity in Test set is {:.4f}'.format(sens_test))    
    print('Specificity in Test set is {:.4f}'.format(spec_test))
    print('Dice/F1 score in Test set is {:.4f}'.format(dice_test))
    print('ROC curve plots saved to ', save_path)
    print('Perf csv saved at ', perf_csv_path)
