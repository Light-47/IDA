from .res_unet_adrian import UNet as unet
from model.UNet_Zoo.UNET import UNet 
import torch

class wnet(torch.nn.Module):
    def __init__(self, n_classes=1, in_c=3, layers=(8,16,32), conv_bridge=True, shortcut=True, mode='train'):
        super(wnet, self).__init__()
        self.unet1 = unet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = unet(in_c=in_c+n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.n_classes = n_classes
        self.mode=mode

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        if self.mode!='train':
            return x2
        return x1,x2

class my_wnet(torch.nn.Module):
    def __init__(self, n_classes=1, in_c=3, mode='train'):
        super(my_wnet, self).__init__()
        self.mode=mode
        self.unet1 = UNet(n_channels=in_c, n_classes=n_classes, mode='aux')  # only output prediction map
        self.unet2 = UNet(n_channels=in_c+n_classes, n_classes=n_classes, mode=self.mode)
        self.n_classes = n_classes
        
    def forward(self, x):
        x1 = self.unet1(x)  # (bs,n_classes,x,x)
        x2, x_encoder, x_mlp = self.unet2(torch.cat([x, x1], dim=1)) # input(bs,in_c+n_classes,x,x), out(bs,n_classes,x,x)
        if self.mode!='train':
            return x2
        return x_encoder, x1, x2, x_mlp