import os
import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from utils.func import adjust_learning_rate
from utils.func import loss_calc
from utils.loss_distri import dice_loss, MPCL, MSEloss2d
from utils.func import mpcl_loss_calc
from utils.mix_branch import Self_CutMix, Asymmetric_Cross_Random_Mix
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import csv
from analyze_results_test import compute_performance
from skimage.transform import resize
from tools.visualize import visualize_img

def iter_eval(model,whole_src_val_loader,whole_trg_val_loader,device,cfg, Iter):
    model.eval()
    model.mode = 'val'
    act = torch.sigmoid if model.n_classes == 1 else torch.nn.Softmax(dim=1)
    # source validation set
    global_auc_s_test, acc_s_test, sens_s_test, spec_s_test, dice_s_test = evaluation(device, model, whole_src_val_loader, act)
    print('######## Source Validation Set ##########')
    print('AUC in Test set is {:.4f}'.format(global_auc_s_test))
    print('Accuracy in Test set is {:.4f}'.format(acc_s_test))
    print('Sensitivity in Test set is {:.4f}'.format(sens_s_test))
    print('Specificity in Test set is {:.4f}'.format(spec_s_test))
    print('Dice/F1 score in Test set is {:.4f}'.format(dice_s_test))
    # target validation set
    global_auc_test, acc_test, sens_test, spec_test, dice_test = evaluation(device, model, whole_trg_val_loader, act)
    print('######## Target Validation Set ##########')
    print('AUC in Test set is {:.4f}'.format(global_auc_test))
    print('Accuracy in Test set is {:.4f}'.format(acc_test))
    print('Sensitivity in Test set is {:.4f}'.format(sens_test))
    print('Specificity in Test set is {:.4f}'.format(spec_test))
    print('Dice/F1 score in Test set is {:.4f}'.format(dice_test))

    array = np.concatenate(([Iter], ['source'], [global_auc_s_test], [acc_s_test], [sens_s_test], [spec_s_test], [dice_s_test], \
                            ['target'], [global_auc_test], [acc_test], [sens_test], [spec_test], [dice_test]), axis=0)
    
    file_path = cfg.TRAIN.SNAPSHOT_DIR + '/data.csv'
    with open(file_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(array)
    # 计算综合指标，粗略评判模型好坏
    metric_src = 0.4 * global_auc_s_test + 0.3 * sens_s_test + 0.3 * dice_s_test
    metric_trg = 0.4 * global_auc_test + 0.3 * sens_test + 0.3 * dice_test
    return metric_src, metric_trg

def label_downsample(device, labels,fea_h,fea_w):

    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda(device)
    labels = F.interpolate(labels, size=fea_w, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()
    labels = F.interpolate(labels, size=fea_h, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()  # n*fea_h*fea_w
    labels = labels.int()
    return labels

def update_class_center_iter(device, cla_src_feas,batch_src_labels,class_center_feas,m):

    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''
    batch_src_feas     = cla_src_feas.detach()
    batch_src_labels   = batch_src_labels.cuda(device)
    n,c,fea_h,fea_w    = batch_src_feas.size()
    batch_y_downsample = label_downsample(device, batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(2):
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda(device)  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c
    class_center_feas = (1-m) * class_center_feas + m * batch_class_center_feas
    return class_center_feas

def generate_prototype_weights(prediction_map, threshold=0.9):
    '''
    利用prediction map生成原型更新权重
    '''
    softmax_pred = torch.softmax(prediction_map, dim=1)
    prob_sort,_  = torch.sort(softmax_pred,dim=1)
    diff_prob    = prob_sort[:,-1]-prob_sort[:,-2]
    weights = torch.sum(diff_prob.ge(threshold).long() == 1).item() / np.size(np.array(diff_prob.cpu()))
    return round(weights,2)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def train_p2hcsl(model, ema_model, whole_src_loader, whole_trg_loader, patch_src_loader, patch_trg_loader, cfg, args):
    '''
        UDA training
    '''
    # create the model and start the training
    src_iter = 0   # best source dice
    trg_iter = 0    # best target dice
    itr = 0         # iter for best source dice
    trg_itr = 0     # iter for best target dice
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    device            = cfg.GPU_ID
    viz_tensorboard   = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    class_center_m    = cfg.TRAIN.CLASS_CENTER_M
    '''init MSE2d for mix_loss'''
    mix_loss = MSEloss2d().cuda(device=device)
    
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMENTATION
    model.train()
    model.cuda(device)
    ema_model.train()
    ema_model.cuda(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    print('finish model setup')

    optimizer = optim.AdamW(model.parameters(), 
                            lr=cfg.TRAIN.LEARNING_RATE,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # compute class center
    '''last channel 1024'''
    if cfg.TRAIN.CLASS_CENTER_FEA_INIT:
        class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    else:
        class_center_feas = np.random.random((2, 1024))
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda(device)
    
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    # whole
    # use whole image to val
    whole_src_loader_iter = enumerate(whole_src_loader)
    whole_trg_loader_iter = enumerate(whole_trg_loader)
    # patch
    patch_src_loader_iter = enumerate(patch_src_loader)
    patch_trg_loader_iter = enumerate(patch_trg_loader)

    mpcl_loss_src = MPCL(device,num_class=cfg.NUM_CLASSES, temperature=cfg.TRAIN.SRC_TEMP,
                                       base_temperature=cfg.TRAIN.SRC_BASE_TEMP, m=cfg.TRAIN.SRC_MARGIN)

    mpcl_loss_trg = MPCL(device, num_class=cfg.NUM_CLASSES, temperature=cfg.TRAIN.TRG_TEMP,
                                       base_temperature=cfg.TRAIN.TRG_BASE_TEMP, m=cfg.TRAIN.TRG_MARGIN)

    ''' source and target test loader'''
    from tools.get_whole_loaders import get_test_dataset
    from torch.utils.data import DataLoader
    data_path = 'data' + '/' + args.csv_target_train.split('/')[-2]
    csv_path = 'val.csv'
    test_dataset = get_test_dataset(data_path, csv_path=csv_path, tg_size=(384,384))
    whole_trg_val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    data_path = 'data' + '/' + args.csv_train.split('/')[-2]
    test_src_dataset = get_test_dataset(data_path, csv_path=csv_path, tg_size=(384,384))
    whole_src_val_loader = DataLoader(test_src_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    # init loss
    Con_loss, mpcl_loss_s2t, mpcl_loss_t2s = 0, 0, 0
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):
        # reset optimizers
        optimizer.zero_grad()
        # adapt LE if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        # UDA training
        '''whole iter'''
        # source
        try:
            _, batch = whole_src_loader_iter.__next__()
        except StopIteration:
            whole_src_loader_iter = enumerate(whole_src_loader)
            _, batch = whole_src_loader_iter.__next__()
        whole_src_images, whole_src_labels = batch
        # target
        try:
            _, batch = whole_trg_loader_iter.__next__()
        except StopIteration:
            whole_trg_loader_iter = enumerate(whole_trg_loader)
            _, batch = whole_trg_loader_iter.__next__()
        whole_trg_images, _ = batch
        '''Patch iter'''
        # source
        try:
            _, batch = patch_src_loader_iter.__next__()
        except StopIteration:
            patch_src_loader_iter = enumerate(patch_src_loader)
            _, batch = patch_src_loader_iter.__next__()
        patch_src_images, patch_src_labels = batch
        # target
        try:
            _, batch = patch_trg_loader_iter.__next__()
        except StopIteration:
            patch_trg_loader_iter = enumerate(patch_trg_loader)
            _, batch = patch_trg_loader_iter.__next__()
        patch_trg_images, _ = batch

        # set train mode for teacher and student models
        model.train()
        ema_model.train() 
        model.mode = 'train'
        if args.source_pretrain:  # 是否进行源域预训练
            '''Self-training Process'''
            if args.self_cutmix: # 是否进行源域cutmix
                loss_seg_s_aux, loss_dice_s_aux, loss_seg_s_main, loss_dice_s_main, p2w_feas = self_training(device, model, 
                                    whole_src_images, whole_src_labels, patch_src_images, patch_src_labels, cfg, args)
            else:
                loss_seg_s_aux, loss_dice_s_aux, loss_seg_s_main, loss_dice_s_main, p2w_feas = self_training_NOCutMix(device, model, whole_src_images, whole_src_labels, cfg)
            if args.self_CL:  # 是否进行源域对比学习            
                class_center_feas = update_class_center_iter(device, p2w_feas[1], p2w_feas[0], class_center_feas,m=class_center_m)
                
                mpcl_loss_t2s = mpcl_loss_calc(device, feas=p2w_feas[2],labels=p2w_feas[0],class_center_feas=class_center_feas,
                                                loss_func=mpcl_loss_src)      
        else:
            # 目标域经过 teacher model 得到 pseudo labels
            # 这里由于只需要利用 ema_model 得到预测结果而不需要其他输出, 因此直接设置 mode='val', 只要不是'train'就可以
            with torch.no_grad():
                ema_model.mode = 'simple_output'  # 只返回预测结果
                pred_trg_whole = ema_model(whole_trg_images.cuda(device))  # whole target
                pred_trg_patch = ema_model(patch_trg_images.cuda(device))  # patch target
                softmax_trg_pred = torch.softmax(pred_trg_whole.detach(), dim=1)
                max_probs, argmax_target_pred = torch.max(softmax_trg_pred, dim=1)            
                softmax_trg_pred_patch = torch.softmax(pred_trg_patch.detach(), dim=1)
                max_probs_patch, argmax_target_pred_patch = torch.max(softmax_trg_pred_patch, dim=1)
            '''Cross-domain Bidirectional Mix'''
            '''生成 Mix后的图像和标签 '''
            t2s_mix, s2t_mix, mask, class_mask = Asymmetric_Cross_Random_Mix(args, [whole_src_images, whole_src_labels], [whole_trg_images, argmax_target_pred.detach().cpu()], 
                                                [patch_src_images, patch_src_labels], [patch_trg_images, argmax_target_pred_patch.detach().cpu()], s_classmix=True)
            ''' Student model: source patch paste to target whole '''
            cla_feas_s2t, pred_s2t_aux, pred_s2t_main, pro_fea_s2t = model(s2t_mix[0].cuda(device))
            ''' Student model: target patch paste to source whole '''
            cla_feas_t2s, pred_t2s_aux, pred_t2s_main, pro_fea_t2s = model(t2s_mix[0].cuda(device))
            '''update class_center using the final layer's features'''
            # ##使用预测图计算原型更新权重
            class_center_m_s2t = generate_prototype_weights(pred_s2t_main.detach(), threshold=0.9)
            class_center_m_t2s = generate_prototype_weights(pred_t2s_main.detach(), threshold=0.9)
            class_center_feas = update_class_center_iter(device, cla_feas_s2t, s2t_mix[1], class_center_feas,m=class_center_m_s2t)
            class_center_feas = update_class_center_iter(device, cla_feas_t2s, t2s_mix[1], class_center_feas,m=class_center_m_t2s)
            # s2t 的 MPCL loss
            mpcl_loss_s2t = mpcl_loss_calc(device, feas=pro_fea_s2t, labels=s2t_mix[1], class_center_feas=class_center_feas,
                                            loss_func=mpcl_loss_trg)
            # t2s 的 MPCL loss
            mpcl_loss_t2s = mpcl_loss_calc(device, feas=pro_fea_t2s,labels=t2s_mix[1],class_center_feas=class_center_feas,
                                            loss_func=mpcl_loss_src)
            ''' 有标签数据使用 CE 和 Dice 损失'''
            # 计算源域 aux, main 分类和分割损失
            mask[0], mask[1] = mask[0].cuda(device), mask[1].cuda(device)
            class_mask[0], class_mask[1] = class_mask[0].cuda(device), class_mask[1].cuda(device)
            patch_src_labels, whole_src_labels = patch_src_labels.cuda(device), whole_src_labels.cuda(device)
            if cfg.TRAIN.MULTI_LEVEL:
                loss_seg_s_aux = cfg.TRAIN.LAMBDA_Patch_AUX * loss_calc(pred_s2t_aux * class_mask[0], (patch_src_labels*class_mask[1]).to(torch.int64), device) + \
                                loss_calc(pred_t2s_aux * (1-mask[0]), (whole_src_labels*(1-mask[1])).to(torch.int64), device) 
                loss_dice_s_aux = cfg.TRAIN.LAMBDA_Patch_AUX * dice_loss(pred_s2t_aux * class_mask[0], (patch_src_labels*class_mask[1]).to(torch.int64), device) + \
                                dice_loss(pred_t2s_aux * (1-mask[0]), (whole_src_labels*(1-mask[1])).to(torch.int64), device)
            else:
                loss_seg_s_aux, loss_dice_s_aux = 0, 0
            loss_seg_s_main = cfg.TRAIN.LAMBDA_Patch_MAIN * loss_calc(pred_s2t_main * class_mask[0], (patch_src_labels*class_mask[1]).to(torch.int64), device) + \
                            loss_calc(pred_t2s_main * (1-mask[0]), (whole_src_labels*(1-mask[1])).to(torch.int64), device) 
            loss_dice_s_main = cfg.TRAIN.LAMBDA_Patch_MAIN * dice_loss(pred_s2t_main * class_mask[0], (patch_src_labels*class_mask[1]).to(torch.int64), device) + \
                            dice_loss(pred_t2s_main * (1-mask[0]), (whole_src_labels*(1-mask[1])).to(torch.int64), device)
                 
            ''' 无标签数据使用一致性损失 '''
            '''目标域分别使用MSE loss, 保证teacher, student网络输出一致'''
            unlabeled_weight_whole = torch.sum(max_probs.ge(0.9).long() == 1).item() / np.size(np.array(max_probs.cpu()))
            unlabeled_weight_patch = torch.sum(max_probs_patch.ge(0.9).long() == 1).item() / np.size(np.array(max_probs_patch.cpu()))
            consistency_weight = 1.0  # Rough, default=1.0
            # MSE的输入 在函数内部处理为 soft label
            Con_loss = consistency_weight * (unlabeled_weight_whole * mix_loss(pred_s2t_main*(1-class_mask[0]), pred_trg_whole*(1-class_mask[0])) + \
                                             unlabeled_weight_patch * mix_loss(pred_t2s_main*mask[0], pred_trg_patch*mask[0]))
        
        '''all'''
        seg_loss = (cfg.TRAIN.LAMBDA_SEG_S_MAIN * loss_seg_s_main
                + cfg.TRAIN.LAMBDA_SEG_S_AUX    * loss_seg_s_aux
                + cfg.TRAIN.LAMBDA_DICE_S_MAIN  * loss_dice_s_main
                + cfg.TRAIN.LAMBDA_DICE_S_AUX   * loss_dice_s_aux
                + cfg.TRAIN.LAMBDA_MPCL_SRC     * mpcl_loss_t2s
                + cfg.TRAIN.LAMBDA_MPCL_TRG     * mpcl_loss_s2t)

        # add consistency loss
        loss = seg_loss + Con_loss

        loss.backward()
        optimizer.step()
        # use ema update ema_model's weight
        if not args.source_pretrain:
            update_model_ema(model, ema_model, 0.99)

        current_losses = {'loss'              : loss,
                          'loss_target_mse'   : Con_loss,   
                          'loss_seg_s_aux'    : loss_seg_s_aux,
                          'loss_seg_s_main'   : loss_seg_s_main,
                          'loss_dice_s_aux'   : loss_dice_s_aux,
                          'loss_dice_s_main'  : loss_dice_s_main,
                          'loss_mpcl_t2s'     : mpcl_loss_t2s,
                          'loss_mpcl_s2t'     : mpcl_loss_s2t,
                          'class_center_m_s2t': class_center_m_s2t,
                          'class_center_m_t2s': class_center_m_t2s
                          }
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir /  f'model_{i_iter}.pth')
            if args.self_CL or not args.source_pretrain:
                # 如果预训练时使用了self_CL 或 不是预训练时就保存类心特征
                class_center_feas_save_dir = cfg.TRAIN.SNAPSHOT_DIR + '/feas'
                os.makedirs(class_center_feas_save_dir,exist_ok=True)
                class_center_feas_save_pth =  f'{class_center_feas_save_dir}/class_center_feas_model_{i_iter}.npy'
                class_center_feas_npy = class_center_feas.cpu().detach().numpy()
                np.save(class_center_feas_save_pth,class_center_feas_npy)

            # 验证
            metric_src, metric_trg = iter_eval(model, whole_src_val_loader, whole_trg_val_loader, device, cfg, i_iter)
            
            if metric_src > src_iter:
                src_iter = metric_src
                itr = i_iter
                print('*'*50)
                print('source present performance: {:.4f}'.format(src_iter))
            if metric_trg > trg_iter:
                trg_iter = metric_trg
                trg_itr = i_iter
                print('target present performance: {:.4f}'.format(trg_iter))
                print('*'*50)

            print('best iter: ', itr)
            print('source best performance: {:.4f}'.format(src_iter))
            print('best iter: ', trg_itr)
            print('target best performance: {:.4f}'.format(trg_iter))

        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

def self_training(device, model, whole_src_images, whole_src_labels, patch_src_images, patch_src_labels, cfg, args):
    # pretrain stage: only mix source images
    p2w_mix, mask, mask_label = Self_CutMix(args, [whole_src_images, whole_src_labels], [patch_src_images, patch_src_labels])
    p2w_feas, pred_p2w_aux, pred_p2w_main, pro_fea_p2w = model(p2w_mix[0].cuda(device))
    mask, mask_label = mask.cuda(device), mask_label.cuda(device)
    whole_src_labels, patch_src_labels = whole_src_labels.cuda(device), patch_src_labels.cuda(device)
    if cfg.TRAIN.MULTI_LEVEL:
        loss_seg_s_aux  = loss_calc(pred_p2w_aux * (1-mask), (whole_src_labels*(1-mask_label)).to(torch.int64), device) + \
                          cfg.TRAIN.LAMBDA_Patch_AUX * loss_calc(pred_p2w_aux * mask, (patch_src_labels*mask_label).to(torch.int64), device) 
        loss_dice_s_aux = dice_loss(pred_p2w_aux * (1-mask), (whole_src_labels*(1-mask_label)).to(torch.int64), device) + \
                          cfg.TRAIN.LAMBDA_Patch_AUX * dice_loss(pred_p2w_aux * mask, (patch_src_labels*mask_label).to(torch.int64), device)
    else:
        loss_seg_s_aux, loss_dice_s_aux = 0, 0
    loss_seg_s_main  = loss_calc(pred_p2w_main * (1-mask), (whole_src_labels*(1-mask_label)).to(torch.int64), device) + \
                       cfg.TRAIN.LAMBDA_Patch_MAIN * loss_calc(pred_p2w_main * mask, (patch_src_labels*mask_label).to(torch.int64), device)
    loss_dice_s_main = dice_loss(pred_p2w_main * (1-mask), (whole_src_labels*(1-mask_label)).to(torch.int64), device) + \
                       cfg.TRAIN.LAMBDA_Patch_MAIN * dice_loss(pred_p2w_main * mask, (patch_src_labels*mask_label).to(torch.int64), device)   
    
    return loss_seg_s_aux, loss_dice_s_aux, loss_seg_s_main, loss_dice_s_main, [p2w_mix[1], p2w_feas, pro_fea_p2w]

def self_training_NOCutMix(device, model, whole_src_images, whole_src_labels, cfg):
    # pretrain stage: only train source whole images
    class_center_feas, pred_aux, pred_main, pro_feas = model(whole_src_images.cuda(device))
    if cfg.TRAIN.MULTI_LEVEL:
        loss_seg_s_aux  = loss_calc(pred_aux, whole_src_labels.to(torch.int64), device)
        loss_dice_s_aux = dice_loss(pred_aux, whole_src_labels.to(torch.int64), device)
    else:
        loss_seg_s_aux, loss_dice_s_aux = 0, 0
    loss_seg_s_main  = loss_calc(pred_main, whole_src_labels.to(torch.int64), device)
    loss_dice_s_main = dice_loss(pred_main, whole_src_labels.to(torch.int64), device)
    
    return loss_seg_s_aux, loss_dice_s_aux, loss_seg_s_main, loss_dice_s_main, [whole_src_labels, class_center_feas, pro_feas]

def evaluation(device, model, whole_val_loader, act):
    all_pred, all_labels = [], []
    with torch.no_grad():
        for images, labels, mask, coords_crop, original_sz, im_list_index in whole_val_loader:
                original_sz = (original_sz[0][0], original_sz[1][0].numpy())  # 转为tuple
                pred_main = model(images.cuda(device))
                pred_main = act(pred_main).squeeze(0)
                pred = pred_main.detach().cpu().numpy()[-1]
                pred = np.expand_dims(resize(pred, output_shape=original_sz, order=3), axis=0) 
                full_pred = np.zeros_like(mask, dtype=float)  # [1,H,W]
                full_pred[:, coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = pred
                full_pred[~np.array(mask).astype(bool)] = 0
                # list存储所有预测结果和标签
                all_pred.append(full_pred) 
                all_labels.append(labels.cpu().numpy())
        # 合并所有预测结果和标签
        all_pred = np.concatenate(all_pred, axis=0)  
        all_labels = np.concatenate(all_labels, axis=0)
        # 展平
        all_pred = all_pred.ravel()
        all_labels = all_labels.ravel()
        # 计算性能指标
        global_auc_test, acc_test, dice_test, _, spec_test, sens_test, _ = \
        compute_performance(all_pred, all_labels, save_path=None, opt_threshold=None)

    return global_auc_test, acc_test, sens_test, spec_test, dice_test


def log_losses_tensorboard(writer,current_losses,i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value),i_iter)

def print_losses(current_losses,i_iter):
    list_strings = []
    for loss_name,loss_value in current_losses.items():
        list_strings.append(f'{loss_name}={to_numpy(loss_value):.3f}')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter={i_iter} {full_string}')

def to_numpy(tensor):
    if isinstance(tensor,(int,float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def train_domain_adaptation(model,ema_model, whole_src_loader, whole_trg_loader, patch_src_loader, patch_trg_loader,cfg, args):
    if cfg.TRAIN.DA_METHOD == 'P2HCSL':
        # patch2whole
        train_p2hcsl(model, ema_model, whole_src_loader, whole_trg_loader, patch_src_loader, patch_trg_loader, cfg, args)
        # here, you can add your own UDA methods
