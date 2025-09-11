import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *


class MSTSGCM(nn.Module):
    def __init__(self):
        super(MSTSGCM, self).__init__()

        self.MSFD = MSFD(in_channels=3, out_channels=32)

        self.ACBlock11 = ResBlockStack1(out_channel=32, num_res=8)
        self.ACBlock12 = ResBlockStack1(out_channel=64, num_res=8)
        self.ACBlock13 = ResBlockStack1(out_channel=128, num_res=8)

        self.bac1 = BasicConv(in_channel=64, out_channel=32, kernel_size=3, stride=1, relu=True, transpose=False)

        self.wtff12 = FeatureFusionNetwork(in_channels=32)

        self.bac2 = BasicConv(in_channel=128, out_channel=64, kernel_size=3, stride=1, relu=True, transpose=False)

        self.wtff23 = FeatureFusionNetwork(in_channels=64)

        self.EAGR1 = EAGR(in_channels=32, out_channels=32)
        self.EAGR2 = EAGR(in_channels=64, out_channels=64)
        self.EAGR3 = EAGR(in_channels=128, out_channels=128)

        self.fam1 = FAM(64)
        self.fam2 = FAM(128)

        self.bac12 = BasicConv(in_channel=32, out_channel=64, kernel_size=3, stride=2, relu=True, transpose=False)
        self.bac12_ = BasicConv(in_channel=64, out_channel=64, kernel_size=3, stride=2, relu=True, transpose=False)
        self.bac23 = BasicConv(in_channel=64, out_channel=128, kernel_size=3, stride=2, relu=True, transpose=False)
        self.bac23_ = BasicConv(in_channel=128, out_channel=128, kernel_size=3, stride=2, relu=True, transpose=False)

        self.ResBlockStack1 = ResBlockStack2(out_channel=32, num_res=8)
        self.ResBlockStack2 = ResBlockStack2(out_channel=64, num_res=8)
        self.ResBlockStack3 = ResBlockStack2(out_channel=128, num_res=8)

        self.bac31 = BasicConv(in_channel=128, out_channel=64, kernel_size=4, stride=2, relu=True, transpose=True)
        self.bac32 = BasicConv(in_channel=128, out_channel=3, kernel_size=3, stride=1, relu=False, transpose=False)


        self.bac21 = BasicConv(in_channel=128, out_channel=32, kernel_size=4, stride=2, relu=True, transpose=True)
        self.bac22 = BasicConv(in_channel=128, out_channel=3, kernel_size=3, stride=1, relu=False, transpose=False)

        self.bac11 = BasicConv(in_channel=64, out_channel=3, kernel_size=3, stride=1, relu=False, transpose=False)


    def forward(self, x, return_frames=False):

        half = F.interpolate(x, scale_factor=0.5)
        quarter = F.interpolate(x, scale_factor=0.25)

        full1, half1, quarter1, fs1, fs2, fs3 = self.MSFD(x)
        full1 = self.EAGR1(x, full1)
        half1 = self.EAGR2(half, half1)
        quarter1 = self.EAGR3(quarter, quarter1)
        ACB1 = self.ACBlock11(full1)  # full 32
        ACB1_2 = F.interpolate(ACB1, scale_factor=0.5)

        ACB2 = self.bac12(ACB1)

        ACB2 = self.fam1(ACB2, half1)

        ACB2 = self.ACBlock12(ACB2)  # half 64
        ACB2_1 = F.interpolate(ACB2, scale_factor=2)
        ACB23 = self.bac23(ACB2)

        ACB3 = self.fam2(ACB23, quarter1)

        ACB3 = self.ACBlock13(ACB3)  # quarter 128

        T3 = F.interpolate(ACB3, scale_factor=2)
        T31 = self.wtff23(ACB1_2, ACB2, T3)

        T2 = F.interpolate(ACB3, scale_factor=4)
        T2 = self.wtff12(ACB1, ACB2_1, T2)

        res1 = self.ResBlockStack1(T2)
        res2 = self.ResBlockStack2(T31)       # 64
        res3 = self.ResBlockStack3(ACB3)     # 128

        bac31 = self.bac31(res3)             # 64
        bac32 = self.bac32(res3) + quarter

        i23 = torch.cat([bac31, res2], dim=1)
        bac21 = self.bac21(i23)
        bac22 = self.bac22(i23) + half

        i12 = torch.cat([bac21, res1], dim=1)
        bac11 = self.bac11(i12) + x

        if return_frames:
            return [self.temporal_deblurring(x, return_intermediate=False, return_frames=True)]

        return [bac32, bac22, bac11, fs1, fs2, fs3]



def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MSTSGM":
        return MSTSGCM()
    raise ModelError('Wrong Model!\nYou should choose MSTSGCM.')

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


