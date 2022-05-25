#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2

high = 0.7
low = 0.3

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x




class FeE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeE, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 3,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


#PD
class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x_temp = self.conv4(x3_2)
        x = self.conv5(x_temp)

        return x

class UnX(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(UnX, self).__init__()
        self.conv1 = BasicConv2d(in_channel,mid_channel,kernel_size=1)
        self.conv2 = BasicConv2d(mid_channel,mid_channel,kernel_size=3,padding=1)
        self.conv3 = BasicConv2d(mid_channel,mid_channel,kernel_size=3,padding=1)
        self.conv4 = BasicConv2d(mid_channel, 1, kernel_size=3, padding=1)

    def forward(self, x, y):
        x_temp = x
        _, dim, _, _ = y.shape

        x = torch.sigmoid(x)
        one = torch.ones_like(x)
        zero = torch.zeros_like(x)
        rever_one = -1 * one
        x = torch.where(x < low, rever_one, x)
        x = torch.where(x > high, zero, x)
        x = torch.where(torch.logical_and(x >= low, x <= high), one, x)
        x = x.expand(-1, dim, -1, -1).mul(y)

        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = self.conv4(x)

        out = out+x_temp

        return out


class OurNet(nn.Module):
    def __init__(self, channel=32):
        super(OurNet, self).__init__()
        # ---- pvt_v2_b2 Backbone ----
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './Pretrained models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # ---- Receptive Field Block like module ----
        self.FeE2 = FeE(128, channel)
        self.FeE3 = FeE(320, channel)
        self.FeE4 = FeE(512, channel)

        # ---- Partial Decoder ----
        self.PD = aggregation(channel)

        # ---- UnX branch 4 ----
        self.UnX4 = UnX(512,256)

        # ---- UnX branch 3 ----
        self.UnX3 = UnX(320,64)

        # ---- UnX branch 2 ----
        self.UnX2 = UnX(128,64)

    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]    #(64,h/2,w/2)
        x2 = pvt[1]    #(128,h/4,w/4)
        x3 = pvt[2]    #(320,h/8,w/8)
        x4 = pvt[3]    #(512,h/16,w/16)

        x2_FeE = self.FeE2(x2)        # channel -> 32
        x3_FeE = self.FeE3(x3)        # channel -> 32
        x4_FeE = self.FeE4(x4)        # channel -> 32

        ra5_feat = self.PD(x4_FeE, x3_FeE, x2_FeE)
        output5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- UnX branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        U4 = self.UnX4(crop_4,x4)
        output4 = F.interpolate(U4, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)


        # ---- UnX branch_3 ----
        crop_3 = F.interpolate(U4, scale_factor=2, mode='bilinear')
        U3 = self.UnX3(crop_3,x3)
        output3 = F.interpolate(U3, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)


        # ---- UnX branch_2 ----
        crop_2 = F.interpolate(U3, scale_factor=2, mode='bilinear')
        U2 = self.UnX2(crop_2,x2)
        output2 = F.interpolate(U2, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return output5, output4, output3, output2




