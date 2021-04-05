from collections import OrderedDict
from .TTVOS_model import *
from models.layers import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def softmax_aggregate(predicted_seg, object_ids):
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in predicted_seg.values()], dim=1).min(dim=1)
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)
    logits = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7)
              for n, seg in [(-1, bg_seg)] + list(predicted_seg.items())}
    logits_sum = torch.cat(list(logits.values()), dim=1).sum(dim=1, keepdim=True)
    aggregated_lst = [logits[n] / logits_sum for n in [-1] + object_ids]
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)
    final_seg_wrongids = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    assert final_seg_wrongids.dtype == torch.int64
    final_seg = torch.zeros_like(final_seg_wrongids)
    for idx, obj_idx in enumerate(object_ids):
        final_seg[final_seg_wrongids == (idx + 1)] = obj_idx
    return final_seg, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in
                       enumerate(object_ids)}


class VOS(nn.Module):
    def __init__(self, backbone, mode='eval', args=None):
        super().__init__()
        self.mode = mode
        refine= args.refine
        LT = args.LT
        self.DiffLoss = args.DiffLoss
        self.TmpLoss = args.TmpLoss

        self.vos = TTVOS(backbone, refine,  LT, args.DiffLoss, args.TmpLoss, args.ltG, args.ltLoc)
        self.TmpLoss_size = int(args.ltLoc.split('s')[-1])
        self.backbone_frozen = False
        self.TmpLoss = args.TmpLoss

    def freeze_backbone(self):
        print("conduct model freeze")
        self.backbone_frozen = True
        #self.vos.backbone.requires_grad(False)
        self.vos.backbone.train(False)
        for param in self.vos.backbone.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)


    def forward(self, x, given_labels=None, state=None, vis_res = None):
        batchsize, nframes, nchannels, prepad_height, prepad_width = x.size()
        seg_lst = []

        if given_labels is not None and not isinstance(given_labels, (tuple, list)):
            given_labels = [given_labels] + (nframes - 1)*[None]

            required_padding = get_required_padding(prepad_height, prepad_width, 16)
            if tuple(required_padding) != (0, 0, 0, 0):
                x, given_labels = apply_padding(x, given_labels, required_padding)
            _, _, _, height, width = x.size()

            video_frames = [elem.view(batchsize, nchannels, height, width) for elem in x.split(1, dim=1)]

            init_label = given_labels[0]
            seg_lst.append(given_labels[0])
            frames_to_process = range(1, nframes)
        else:
            required_padding = get_required_padding(prepad_height, prepad_width, 16)
            if tuple(required_padding) != (0, 0, 0, 0):
                x, given_labels = apply_padding(x, given_labels, required_padding)
            _, _, _, height, width = x.size()

            video_frames = [elem.view(batchsize, nchannels, height, width) for elem in x.split(1, dim=1)]
            zero_label = torch.zeros_like(x)[0,0,0,:,:].unsqueeze(0).unsqueeze(0).long()
            idx=0
            for this_l in given_labels:
                idx+=1
                if this_l !=None:
                    init_label =this_l
                    seg_lst.append(this_l)
                    break
                else:
                    given_labels[idx] = zero_label
                    seg_lst.append(zero_label)
            frames_to_process = range(idx, nframes)

        object_ids = init_label.unique().tolist()

        if 0 in object_ids:
            object_ids.remove(0)
        state = {}
        if self.DiffLoss and self.training:
            prev_seg_tensor = torch.zeros(size=(batchsize, nframes - 1, 2, height, width)).to(DEVICE)
        if self.TmpLoss and self.training:
            Tmp_seg_tensor = torch.zeros(size=(batchsize, nframes - 1, 1, height, width)).to(DEVICE)


        for obj_idx in object_ids:
            given_seg = torch.cat([init_label != obj_idx, init_label == obj_idx], dim=-3).float()
            state[obj_idx] = self.vos.get_init_state(video_frames[0], given_seg)
            if self.DiffLoss and self.training:
                prev_seg_tensor[:, 0, :, :, :] = given_seg
            if self.TmpLoss and self.training:
                Tmp_seg_tensor[:, 0, :, :, :] = (init_label == obj_idx).float()

        logseg_lsts = {k: [] for k in object_ids}            # Used for training
        Refine_heatmap_lsts = {k: [] for k in object_ids}

        vis_info = {k: (nframes - 1) * [None] for k in object_ids}
        Tmp_seg_lsts={k: [] for k in object_ids}

        for k in object_ids:
            tmp = (given_labels[0] == k).float().clamp(1e-7, 1 - 1e-7)
            tmp = torch.cat([1 - tmp, tmp], dim=1)
            logseg_lsts[k].append(tmp.log())

            if self.TmpLoss:
                tmp = (given_labels[0] == k).float()
                Tmp_seg_lsts[k].append(F.avg_pool2d(tmp, self.TmpLoss_size))

        for i in frames_to_process:
            if not self.training and given_labels[i] is not None:
                seg_lst.append(given_labels[i])
                temp_object_ids = torch.unique(given_labels[i]).unique().tolist()
                if 0 in temp_object_ids:
                    temp_object_ids.remove(0)
                temp_ = torch.ones_like(logseg_lsts[object_ids[0]][0])
                temp_[0,0,:,:] = ((1-1e-7)*temp_[0,0,:,:]).log()
                temp_[0,1,:,:] = (1e-7*temp_[0,1,:,:]).log()

                new_seg_lst = []
                for obj_idx in temp_object_ids:
                    given_seg = torch.cat([given_labels[i] != obj_idx, given_labels[i] == obj_idx], dim=-3).float()
                    state[obj_idx] = self.vos.get_init_state(video_frames[0], given_seg)
                    if self.DiffLoss and self.training:
                        prev_seg_tensor[:, 0, :, :, :] = given_seg
                    for j in range(0, i):
                        new_seg_lst.append(temp_)
                        logseg_lsts.update({obj_idx: new_seg_lst})

                object_ids += temp_object_ids

            else:
                feats = self.vos.extract_feats(video_frames[i])
                segscore = {}
                segscore_coarse = {}
                for k in object_ids:
                    #print("State : " + str(state[k]["prev_seg"].shape))
                    # self.vos(feats, state[k])
                    state[k], segscore[k], segscore_coarse[k] = self.vos(feats, state[k])
                    if self.training and self.DiffLoss:
                        Refine_heatmap_lsts[k].append(F.softmax(self.vos.refine.Diff_heatmap, dim=-3))
                    if self.training and self.TmpLoss:
                        Tmp_seg_lsts[k].append(self.vos.refine.Tmp_heatmap)

                predicted_seg = {k: F.softmax(segscore[k], dim=-3) for k in object_ids}

                if vis_res is not None:
                    for k in object_ids:
                        temp=[]
                        temp.append(state[k]['visual_Gtensor'])
                        temp.append(state[k]['visual_Gname'])
                        temp.append(state[k]['visual_Itensor'])
                        temp.append(state[k]['visual_Iname'])
                        temp.append(state[k]['visual_tensor'])
                        temp.append(state[k]['visual_name'])
                        vis_info[k][i - 1] = temp

                if self.training:
                    update_seg = {k: predicted_seg[k] for k in object_ids}
                    if self.DiffLoss and i <nframes-1:
                        prev_seg_tensor[:,i,:,:,:] = update_seg[1]

                else:
                    output_seg, aggregated_seg = softmax_aggregate(predicted_seg, object_ids)
                    update_seg = {n: aggregated_seg[n] for n in object_ids}
                # if i < frames_to_process[-1]:
                for k in object_ids:
                    state[k] = self.vos.update(feats, update_seg[k], state[k], i)

                # Training
                for k in object_ids:
                    logseg_lsts[k].append(F.log_softmax(segscore[k], dim=-3))

                if not self.training:
                    seg_lst.append(output_seg)

        output = {}
        if not self.training:
            output['segs'] = torch.stack(seg_lst, dim=1)
            output['segs'] = unpad(output['segs'], required_padding)
            if vis_res is not None:
                output['segs_vis'] = vis_info

        output['logsegs'] = {k: torch.stack(logseg_lsts[k], dim=1) for k in object_ids}
        output['logsegs'] = unpad(output['logsegs'], required_padding)

        if self.TmpLoss:
            output['Tmp_result'] = {k: torch.stack(Tmp_seg_lsts[k], dim=1) for k in object_ids}
            output['Tmp_result'] = unpad(output['Tmp_result'], required_padding)

        if self.training and self.DiffLoss :
            output['prev_seg_tensor'] = prev_seg_tensor
            output['Diff_heat'] = {k: torch.stack(Refine_heatmap_lsts[k], dim=1) for k in object_ids}

        return output, state
