import torch
import random

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred, classes.unsqueeze(0).unsqueeze(1))
    N = pred.eq(classes)
    return N

def Self_CutMix(args, whole, patch):
    _, _, W, H = whole[0].size()
    half_size = args.concat_size // 2
    # 创建全0 mask
    mask = torch.zeros(1, 1, W, H)
    mask_label = torch.zeros(1, W, H)
    # 随机初始化一个拼接中心，大小为128, 128
    center_x = random.randint(half_size, W-half_size)  # set boundery 
    center_y = random.randint(half_size, H-half_size)
    mask[:, :, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    mask_label[:, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    # copy-paste imgs and labels
    p2w_mix_imgs   = whole[0] * (1-mask) + patch[0] * mask
    p2w_mix_labels = whole[1] * (1-mask_label) + patch[1] * mask_label
    return [p2w_mix_imgs, p2w_mix_labels], mask, mask_label


def Asymmetric_Cross_Random_Mix(args, whole_src, whole_trg, patch_src, patch_trg, s_classmix=True):
    '''
    非对称结构： \n
    t2s: 目标域的patch copy-paste 到源域whole上\n
    s2t: 源域patch中的 foreground copy-paste 到目标域whole上
    '''
    _, _, W, H = whole_src[0].size()
    half_size = args.concat_size // 2
    # 创建全0 mask
    mask = torch.zeros(1, 1, W, H)
    mask_label = torch.zeros(1, W, H)
    # 随机初始化一个拼接中心，大小为128, 128
    center_x = random.randint(half_size, W-half_size)  # set boundery 
    center_y = random.randint(half_size, H-half_size)
    mask[:, :, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    mask_label[:, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    if s_classmix:  # source classmix TO target domain
        # Bidirectional copy-paste imgs and labels
        t2s_mix_imgs   = whole_src[0] * (1-mask) + patch_trg[0] * mask
        t2s_mix_labels = whole_src[1] * (1-mask_label) + patch_trg[1] * mask_label
        # patch classmix
        ClassMask_label = generate_class_mask(patch_src[1]*mask_label, classes=torch.tensor(1)).float()  # [bs, w, h]
        ClassMask  = ClassMask_label.unsqueeze(1)  # [bs, 1, w, h]
        s2t_mix_imgs   = whole_trg[0] * (1-ClassMask) + patch_src[0] * ClassMask
        s2t_mix_labels = whole_trg[1] * (1-ClassMask_label) + patch_src[1] * ClassMask_label
        return [t2s_mix_imgs,t2s_mix_labels], [s2t_mix_imgs,s2t_mix_labels], [mask, mask_label], [ClassMask, ClassMask_label]
    else:
        # Bidirectional copy-paste imgs and labels
        s2t_mix_imgs   = whole_trg[0] * (1-mask) + patch_src[0] * mask
        s2t_mix_labels = whole_trg[1] * (1-mask_label) + patch_src[1] * mask_label
        # patch classmix
        ClassMask_label = generate_class_mask(patch_trg[1]*mask_label, classes=torch.tensor(1)).float()  # [bs, w, h]
        ClassMask  = ClassMask_label.unsqueeze(1)  # [bs, 1, w, h]
        t2s_mix_imgs   = whole_src[0] * (1-ClassMask) + patch_trg[0] * ClassMask
        t2s_mix_labels = whole_src[1] * (1-ClassMask_label) + patch_trg[1] * ClassMask_label
        return [t2s_mix_imgs,t2s_mix_labels], [s2t_mix_imgs,s2t_mix_labels], [ClassMask, ClassMask_label], [mask, mask_label]

def Symmetric_Cross_Random_CutMix(args, whole_src, whole_trg, patch_src, patch_trg):
    '''
    对称结构： \n
    t2s: 目标域的patch copy-paste 到源域 whole上\n
    s2t:   源域的patch copy-paste 到目标域 whole上
    '''
    _, _, W, H = whole_src[0].size()
    half_size = args.concat_size // 2
    # 创建全0 mask
    mask = torch.zeros(1, 1, W, H)
    mask_label = torch.zeros(1, W, H)
    # 随机初始化一个拼接中心，大小为128, 128
    center_x = random.randint(half_size, W-half_size)  # set boundery 
    center_y = random.randint(half_size, H-half_size)
    mask[:, :, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    mask_label[:, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    # Bidirectional copy-paste imgs and labels
    t2s_mix_imgs   = whole_src[0] * (1-mask) + patch_trg[0] * mask
    t2s_mix_labels = whole_src[1] * (1-mask_label) + patch_trg[1] * mask_label
    s2t_mix_imgs   = whole_trg[0] * (1-mask) + patch_src[0] * mask
    s2t_mix_labels = whole_trg[1] * (1-mask_label) + patch_src[1] * mask_label
    return [t2s_mix_imgs,t2s_mix_labels], [s2t_mix_imgs,s2t_mix_labels], [mask, mask_label]

def Symmetric_Cross_Random_ClassMix(args, whole_src, whole_trg, patch_src, patch_trg):
    '''
    对称结构： \n
    t2s: 目标域的patch copy-paste 到源域 whole上\n
    s2t:   源域的patch copy-paste 到目标域 whole上
    '''
    _, _, W, H = whole_src[0].size()
    half_size = args.concat_size // 2
    # 创建全0 mask
    mask = torch.zeros(1, 1, W, H)
    mask_label = torch.zeros(1, W, H)
    # 随机初始化一个拼接中心，大小为128, 128
    center_x = random.randint(half_size, W-half_size)  # set boundery 
    center_y = random.randint(half_size, H-half_size)
    mask[:, :, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    mask_label[:, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    # Bidirectional copy-paste imgs and labels
    ClassMask_s_label = generate_class_mask(patch_src[1]*mask_label, classes=torch.tensor(1)).float()  # [bs, w, h]
    ClassMask_s  = ClassMask_s_label.unsqueeze(1)  # [bs, 1, w, h]
    s2t_mix_imgs   = whole_trg[0] * (1-ClassMask_s) + patch_src[0] * ClassMask_s
    s2t_mix_labels = whole_trg[1] * (1-ClassMask_s_label) + patch_src[1] * ClassMask_s_label

    ClassMask_t_label = generate_class_mask(patch_trg[1]*mask_label, classes=torch.tensor(1)).float()  # [bs, w, h]
    ClassMask_t  = ClassMask_t_label.unsqueeze(1)  # [bs, 1, w, h]
    t2s_mix_imgs   = whole_src[0] * (1-ClassMask_t) + patch_trg[0] * ClassMask_t
    t2s_mix_labels = whole_src[1] * (1-ClassMask_t_label) + patch_trg[1] * ClassMask_t_label    
    return [t2s_mix_imgs,t2s_mix_labels], [s2t_mix_imgs,s2t_mix_labels], [ClassMask_s, ClassMask_s_label], [ClassMask_t, ClassMask_t_label]

def Uni_Cross_Random_Mix(args, whole_src, whole_trg, patch_src, patch_trg, s2t=True, cutmix=True):
    '''
    单边结构： \n
    S2T means 源域patch拼到目标域上, 源域SelfMix \n
    T2S means 目标域patch拼到源域上, 目标域SelfMix
    '''
    _, _, W, H = whole_src[0].size()
    half_size = args.concat_size // 2
    # 创建全0 mask
    mask = torch.zeros(1, 1, W, H)
    mask_label = torch.zeros(1, W, H)
    # 随机初始化一个拼接中心，大小为128, 128
    center_x = random.randint(half_size, W-half_size)  # set boundery 
    center_y = random.randint(half_size, H-half_size)
    mask[:, :, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    mask_label[:, center_x-half_size:center_x+half_size, center_y-half_size:center_y+half_size] = 1
    if s2t:
        if cutmix:
            s2t_mix_imgs   = whole_trg[0] * (1-mask) + patch_src[0] * mask
            s2t_mix_labels = whole_trg[1] * (1-mask_label) + patch_src[1] * mask_label
            s2s_mix_imgs   = whole_src[0] * (1-mask) + patch_src[0] * mask
            s2s_mix_labels = whole_src[1] * (1-mask_label) + patch_src[1] * mask_label
            return [s2s_mix_imgs,s2s_mix_labels], [s2t_mix_imgs,s2t_mix_labels], [mask, mask_label]            
        else:
            ClassMask_label = generate_class_mask(patch_src[1]*mask_label, classes=torch.tensor(1)).float()  # [bs, w, h]
            ClassMask  = ClassMask_label.unsqueeze(1)  # [bs, 1, w, h]
            s2t_mix_imgs   = whole_trg[0] * (1-ClassMask) + patch_src[0] * ClassMask
            s2t_mix_labels = whole_trg[1] * (1-ClassMask_label) + patch_src[1] * ClassMask_label
            s2s_mix_imgs   = whole_src[0] * (1-ClassMask) + patch_src[0] * ClassMask
            s2s_mix_labels = whole_src[1] * (1-ClassMask_label) + patch_src[1] * ClassMask_label
            return [s2s_mix_imgs,s2s_mix_labels], [s2t_mix_imgs,s2t_mix_labels], [ClassMask, ClassMask_label]
    else:
        if cutmix:
            t2s_mix_imgs   = whole_src[0] * (1-mask) + patch_trg[0] * mask
            t2s_mix_labels = whole_src[1] * (1-mask_label) + patch_trg[1] * mask_label
            t2t_mix_imgs   = whole_trg[0] * (1-mask) + patch_trg[0] * mask
            t2t_mix_labels = whole_trg[1] * (1-mask_label) + patch_trg[1] * mask_label
            return [t2s_mix_imgs,t2s_mix_labels], [t2t_mix_imgs,t2t_mix_labels], [mask, mask_label]
        else:
            ClassMask_label = generate_class_mask(patch_trg[1]*mask_label, classes=torch.tensor(1)).float()  # [bs, w, h]
            ClassMask  = ClassMask_label.unsqueeze(1)  # [bs, 1, w, h]
            t2s_mix_imgs   = whole_src[0] * (1-ClassMask) + patch_trg[0] * ClassMask
            t2s_mix_labels = whole_src[1] * (1-ClassMask_label) + patch_trg[1] * ClassMask_label
            t2t_mix_imgs   = whole_trg[0] * (1-ClassMask) + patch_trg[0] * ClassMask
            t2t_mix_labels = whole_trg[1] * (1-ClassMask_label) + patch_trg[1] * ClassMask_label            
            return [t2s_mix_imgs,t2s_mix_labels], [t2t_mix_imgs,t2t_mix_labels], [ClassMask, ClassMask_label]

