import torch
import torch.nn as nn

'''Depthwise Separable Convolution MLP'''
class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(channel, channel)
        self.depthwise = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=channel, bias=False)
        self.pointwise = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, bias=False)
        # self.act = nn.ReLU()
        self.act = nn.GELU()
        self.drop = nn.Dropout(0)

    def forward(self, x):
        x_conv = self.pointwise(self.depthwise(x))
        x_fc = self.fc1(x.permute(0, 2, 3, 1))
        x_fc = x_fc.permute(0, 3, 1, 2)
        x_mlp = self.drop(self.act(x_fc + x_conv))
        # x_mlp = nn.functional.normalize(x_mlp, p=2, dim=1)
        return x_mlp

class Rectify_MLP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Rectify_MLP, self).__init__()
        self.fc1 = nn.Linear(in_channel, out_channel)
        self.depthwise = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1,
                                   padding=1, dilation=1, groups=in_channel, bias=False)
        self.pointwise = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0)

    def forward(self, x):
        x_conv = self.pointwise(self.depthwise(x))
        x_fc   = self.fc1(x.permute(0, 2, 3, 1))
        x_fc   = x_fc.permute(0, 3, 1, 2)
        x_mlp = self.drop(self.act(x_fc + x_conv))
        return x_mlp

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

class ProjectionHead(nn.Module):
    def __init__(self, channel):
        super(ProjectionHead, self).__init__()
        self.ProjHead = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel // 2, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        # x_proj = self.ProjHead(x)
        x_proj = nn.functional.normalize(self.ProjHead(x), p=2, dim=1)
        return x_proj


