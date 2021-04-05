import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class Boundary_binaryLoss(torch.nn.Module):
    def __init__(self, kernel_size = 15, ignore_index=255, Nreturn_gt=True, criteria = nn.NLLLoss):

        super().__init__()
        self.kernel_size = kernel_size
        self.ignore = ignore_index
        self.criteria= criteria(ignore_index=ignore_index)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.Nreturn_gt = Nreturn_gt


    def Define_Boundary_region(self, label):
        """
        Define new ground truth
        change gt value of not boudary region into self.ignore
        """
        if torch.cuda.is_available():
            np_label = label.data.cpu().numpy().astype(np.uint8)
        else:
            np_label = label.data.numpy().astype(np.uint8)

        orglabel= np_label.copy()
        np_label *= 255
        np_label[orglabel==self.ignore]=0 #  v=ignore => 0

        erosion = cv2.erode(np_label, self.kernel, iterations=1)
        dilation = cv2.dilate(np_label, self.kernel, iterations=1)
        boundary = dilation - erosion
        edgemap = 255 * torch.ones_like(label)
        edgemap[torch.from_numpy(boundary) > 0] = label[torch.from_numpy(boundary) > 0] # 255, 0, 1, ignore
        edgemap[torch.from_numpy(orglabel==self.ignore)] = 255 # 255, 0, 1, ignore => 255


        return edgemap

    def forward(self, logits, labels):
        """
        Boundary_binaryLoss
          logits: [B, C, H, W] Variable, estimated logits at each pixel
          labels: [B, H, W] Tensor, binary ground truth masks (0 , 1, ignore)

        """
        ph, pw = logits.size(2), logits.size(3)
        h, w = labels.size(1), labels.size(2)
        if ph != h or pw != w:
            logits = F.upsample(input=logits, size=(h, w), mode='bilinear')
        boundary_label = torch.zeros_like(labels)
        for i in range(labels.size(0)):
            boundary_label[i,:,:] = self.Define_Boundary_region(labels[i,:,:])

        loss = self.criteria(logits,boundary_label.long())
        if self.Nreturn_gt:
            return loss
        else:
            return loss, boundary_label

class OhemNLLLoss(nn.Module):
    def __init__(self, ignore_index=-1, thres=0.7,
        min_kept_ratio=0.7, weight=None):
        super(OhemNLLLoss, self).__init__()
        self.thresh = thres
        self.min_kept_ratio = min_kept_ratio
        self.ignore_index = ignore_index
        self.criterion = nn.NLLLoss(weight=weight,
                                    ignore_index=ignore_index,
                                    reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        min_kept = int(self.min_kept_ratio * h * w)
        if ph != h or pw != w:
            print("bilinear")
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

class OhemNLLEdgeLoss(nn.Module):
    def __init__(self, ignore_index=-1, threshold=0.7, min_kept_ratio=0.7, weight=None, edge_loss_scale=0.5):
        super(OhemNLLEdgeLoss, self).__init__()
        self.ohemLoss = OhemNLLLoss(ignore_index=ignore_index, thres=threshold,
            min_kept_ratio=min_kept_ratio, weight=weight)
        self.edgeLoss = Boundary_binaryLoss(ignore_index=ignore_index)
        self.edge_loss_scale = edge_loss_scale

    def forward(self, score, target):
        loss = self.ohemLoss(score, target) + self.edge_loss_scale * self.edgeLoss(score, target)
        return(loss)


class NLLLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, weight=None, ignore_index = None, align=True):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        '''

        super().__init__()

        self.loss = nn.NLLLoss(weight, ignore_index=ignore_index)
        self.align = align

    def forward(self, outputs, targets):
        # print(torch.unique(targets))
        ph, pw = outputs.size(2), outputs.size(3)
        h, w = targets.size(1), targets.size(2)

        if ph != h or pw != w:
            outputs= F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=self.align)

        return self.loss(outputs, targets)