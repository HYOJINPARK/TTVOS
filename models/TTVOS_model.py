
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import math
from models.layers import *
from models.Dnc_models import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


REFINE_MAP = {
    'v3': REFINEV3,
}

class GlobalContextMemoryMV1(nn.Module):
    def __init__(self, ch_in, ch_key, ch_val, group_n=4):
        super().__init__()
        self.conv_key = nn.Sequential(
            nn.Conv2d(ch_in, ch_key, 1, 1, 0),
            nn.Conv2d(ch_key, ch_key, 5, 1, 2, groups=ch_key//group_n),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_val = nn.Sequential(
            nn.Conv2d(ch_in, ch_val, 1, 1, 0),
            nn.Conv2d(ch_val, ch_val, 5, 1, 2, groups=ch_val//group_n),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.scale_constant = 1./math.sqrt(ch_key)

    def forward(self, input, mask=None):
        if mask !=None:
            input = torch.cat([input, mask], dim=1)
        key = self.conv_key(input)
        val = self.conv_val(input)
        B, KC, H, W = key.shape
        B, KV, H, W = val.shape
        key_mm = key.reshape(B, KC, -1)
        val_mm = val.reshape(B, KV, -1).permute((0, 2, 1)).contiguous()
        out = F.softmax(torch.bmm(key_mm, val_mm).mul(1./math.sqrt(H*W)), dim=2)
        return(out)




class GlobalContextQueryMV1(nn.Module):
    def __init__(self, ch_in, ch_val, ch_feat, group_n=4):
        super().__init__()

        self.conv_val = nn.Sequential(
            nn.Conv2d(ch_in, ch_val, 1, 1, 0),
            nn.Conv2d(ch_val, ch_val, 5, 1, 2, groups=ch_val//group_n),
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.conv_feat = nn.Sequential(
            nn.Conv2d(ch_in, ch_feat, 1, 1, 0),
            nn.Conv2d(ch_feat, ch_feat, 5, 1, 2, groups=ch_feat//group_n),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.blend = nn.Sequential(
            nn.Conv2d(2 * ch_feat, 2 * ch_feat, 5, 1, 2, groups=ch_feat//2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * ch_feat, ch_feat, 1, 1, 0)
        )

    def forward(self, input, gc):
        val = self.conv_val(input)
        feat = self.conv_feat(input)
        if gc is None:
            gc_act = torch.zeros_like(feat)
        else:
            B, K, H, W = val.shape
            val_mm = val.reshape(B, K, -1)

            gc_act = torch.bmm(gc, val_mm).reshape(B, -1, H, W)
        out = F.relu(self.blend(torch.cat([feat, gc_act], dim=1)))
        return(out)


def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch*channel, padding=(kernel.size(2)-1)//2)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class Matrix_Multiplicaiotn(nn.Module):
    def __init__(self):
        super(Matrix_Multiplicaiotn, self).__init__()

    def forward(self, A, B):
        b, c_k, h, w = A.size()
        _, c_v, _, _ = B.size()
        mi = A.view(b, c_k, h * w)  # b, emb1, HW
        qi = B.view(b, c_v, h * w)  # b, HW, emb2
        qi = torch.transpose(qi, 1, 2)

        # print("mi : " +str(mi.size()))
        # print("qi : " +str(qi.size()))

        C = torch.bmm(mi, qi)  # b, emb1, emb2
        return C



class TTVOS(nn.Module):
    def __init__(self, backbone_cfg, refine='v3', LT="Not", DiffLoss=False, Tmp_loss=False, ltG=4, LT_loc='s8'):
        super().__init__()
        self.backbone = getattr(models.backbones, backbone_cfg[0])(*backbone_cfg[1])
        channel_counts = self.backbone.get_feature_count()
        self.refine_N=refine
        refine_func = REFINE_MAP[refine]
        self.refine = refine_func(ch_s16=channel_counts[-3], ch_s8=channel_counts[-2], ch_s4=channel_counts[-1],
                                  DiffLoss=DiffLoss, Tmp_loss=Tmp_loss)
        self.addCoords = AddCoords(with_r=True)

        self.LT = LT
        self.LT_loc = LT_loc

        if LT_loc=='s8':
            this_feature = channel_counts[-2]
        else:
            this_feature = channel_counts[-3]

        self.gcm = GlobalContextMemoryMV1(this_feature + 9, channel_counts[-2], channel_counts[-2])
        self.gcq = GlobalContextQueryMV1(this_feature + 9, channel_counts[-2], channel_counts[-2], group_n=ltG)


    def get_init_state(self, img, given_seg):
        state = {}
        pool_N = 4  if self.LT_loc =='s16' else  3

        for _ in range(pool_N):
            given_seg = F.avg_pool2d(given_seg, kernel_size=3, stride=2, padding=1)
        temp_seg = torch.zeros_like(given_seg)
        prev_seg = torch.cat([temp_seg, temp_seg, temp_seg],dim=1)
        prev_seg[:,:2,:,:]=given_seg
        state['prev_seg'] = prev_seg

        if self.gcq is not None:
            feats = self.backbone.get_features(img)
            curr_lt = self.gcm(self.addCoords(feats[self.LT_loc]), prev_seg)

            state['gc'] = curr_lt
        else:
            feats = self.backbone.get_features(img)


        state['prev_seg_feats16'] = feats['s16']

        return state


    def update(self, feats, pred_seg, state, t):

        if pred_seg is not None:
            if self.LT_loc == 's16':
                for _ in range(4):
                    pred_seg = F.avg_pool2d(pred_seg, kernel_size=3, stride=2, padding=1)
            else:
                for _ in range(3):
                    pred_seg = F.avg_pool2d(pred_seg, kernel_size=3, stride=2, padding=1)

        temp_seg = torch.zeros_like(state['prev_seg'])
        temp_seg[:,:2,:,:] = pred_seg
        temp_seg[:,2:4,:,:] = state['prev_seg'][:,:2,:,:]
        temp_seg[:,4:,:,:] = state['prev_seg'][:,2:4,:,:]

        state['prev_seg'] = temp_seg
        state['prev_seg_feats16'] = feats['s16']

        if (self.gcm is not None) and (pred_seg is not None): #update long-term
            curr_gc = self.gcm(self.addCoords(feats[self.LT_loc]), temp_seg)
            state['gc'] = state['gc']*(1/(state['update_W']+1)) + curr_gc *(state['update_W']/(state['update_W']+1))

        return state

    def extract_feats(self, img):
        feats = self.backbone.get_features(img)
        return feats

    def forward(self, feats, state):
        return_feature = list(feats.values())

        if self.gcq is not None:
            this_feat = self.addCoords(feats[self.LT_loc])
            this_feat = torch.cat([this_feat, state['prev_seg']], 1)
            gcp_feat = self.gcq(this_feat, state['gc'])

            if self.LT_loc=='s16' :
                return_feature = (gcp_feat, feats['s8'], feats['s4'])
            else:
                return_feature= (feats['s16'], gcp_feat, feats['s4'])

        segscore, segscore_coarse = self.refine(return_feature, state)

        if self.training == False:
            # self.visual_Glist, self.visual_Glist = [], []
            state['visual_Gtensor'] = self.refine.visual_Gtensor
            state['visual_Gname'] = self.refine.visual_Gname
            state['visual_Itensor'] = self.refine.visual_Itensor
            state['visual_Iname'] = self.refine.visual_Iname
            state['visual_tensor'] = self.refine.visual_tensor
            state['visual_name'] = self.refine.visual_name
        return state, segscore, segscore_coarse

