
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import math
from models.layers import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('naf', nn.ReLU(inplace=True))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




class REFINEV3(nn.Module):
    def __init__(self, ch_s16=2048, ch_s8=512, ch_s4=256, DiffLoss=False, Tmp_loss=False):
        super().__init__()
        ch_out_inter = min(128, ch_s4)
        ch_out_s16 = min(2 * ch_out_inter, ch_s16)
        ch_out_deconv = 2
        self.DiffLoss = DiffLoss
        self.Tmp_loss = Tmp_loss


        self.conv_s16 = ConvRelu(2*3 + ch_s16, ch_out_s16, 3, 1, 1)
        self.blend_s16 = ConvRelu(ch_out_s16 + ch_s16, ch_out_inter, 3, 1, 1)
        self.conv_s8 = ConvRelu(ch_s8, ch_out_inter, 1, 1, 0)
        self.blend_s8 = ConvRelu(ch_out_inter + ch_out_deconv, ch_out_inter, 3, 1, 1)
        self.cal_UpW = nn.Sequential(
            ConvRelu(ch_out_inter + ch_out_deconv, (ch_out_inter + ch_out_deconv)//2, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d( (ch_out_inter + ch_out_deconv)//2, 1, 1,1,0),
            nn.Sigmoid()
        )
        self.conv_s4 = ConvRelu(ch_s4, ch_out_inter, 1, 1, 0)
        self.blend_s4 = ConvRelu(ch_out_inter + ch_out_deconv, ch_out_inter, 3, 1, 1)
        self.deconv1_1 = nn.ConvTranspose2d(ch_out_inter, ch_out_deconv, 4, 2, 1, bias=True)
        self.deconv1_2 = nn.ConvTranspose2d(ch_out_deconv, ch_out_deconv, 4, 2, 1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(ch_out_inter, ch_out_deconv, 4, 2, 1, bias=True)
        self.deconv3 = nn.ConvTranspose2d(ch_out_inter + 2 * ch_out_deconv, 2 * 4, 4, 2, 1, bias=True)
        self.predictor = nn.PixelShuffle(2)

        self.predictor_Diff = nn.Conv2d(ch_out_inter, 2, 3, 1, 1)
        self.predictor_Tmp = nn.Conv2d(ch_s8, 1, 3, 1, 1)


    def forward(self, feats, state):
        s16, LT_sim, s4 = feats

        prev_seg = F.avg_pool2d(state['prev_seg'], 3, 2, 1)
        conv_s16_prev = state['prev_seg_feats16']
        shortTmp = self.conv_s16(torch.cat([prev_seg, conv_s16_prev], dim=1))

        u16 = self.blend_s16(torch.cat([s16, shortTmp], dim=1))
        out_16 = self.deconv1_1(u16)

        u8 = torch.cat([self.conv_s8(LT_sim), out_16], dim=-3)
        state['update_W'] = self.cal_UpW(u8).squeeze(3).expand(LT_sim.size(0),LT_sim.size(1),LT_sim.size(1))

        u8 = self.blend_s8(u8)
        out_8 = self.deconv2(u8)
        segscore_coarse = out_8

        u = torch.cat([self.conv_s4(s4), out_8], dim=-3)
        out_4 = self.blend_s4(u)
        out_4 = self.deconv3(torch.cat([self.deconv1_2(out_16), out_8, out_4], dim=1))

        segscore = self.predictor(out_4)

        if self.training:
            if self.DiffLoss:
                self.Diff_heatmap = self.predictor_Diff(u8)

            if self.Tmp_loss:
                self.Tmp_heatmap = torch.sigmoid(self.predictor_Tmp(LT_sim))

        else:
            self.visual_Gtensor, self.visual_Gname = None, None
            # if self.DiffLoss:
            #     self.visual_Gtensor, self.visual_Gname = [], []
            #     self.Diff_heatmap = self.predictor_Diff(u8)
            #     self.visual_Gtensor.append(self.Diff_heatmap)
            #     self.visual_Gname.append("Diff_heatmap")

            self.visual_Itensor, self.visual_Iname = None, None


            self.visual_tensor, self.visual_name = None, None

            # if self.Tmp_loss:
            #     self.visual_tensor, self.visual_name = [], []
            #     self.Tmp_heatmap = torch.sigmoid(self.predictor_Tmp(s8))
            #     self.visual_tensor.append(self.Tmp_heatmap[0])
            #     self.visual_name.append("Tmp_heatmap")

        return segscore, segscore_coarse

