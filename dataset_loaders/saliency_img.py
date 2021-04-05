
import numpy as np
import glob
import os

from PIL import Image
import torch
import torchvision as tv

import random
from dataset_loaders.custom_transforms import aug_batch, aug_tps
from dataset_loaders.dataset_utils import train_image_post_read, train_label_post_read
import cv2

class SaliecyDatasetVid(torch.utils.data.Dataset):
    def __init__(self, root_path, image_read=None, anno_read=None, use_tps=True,color_aug=True, sample_len=3, size=(240, 432),
                nb_points=5):

        self._root_path = root_path
        self._image_read = image_read
        self._anno_read = anno_read
        self._size = size
        self._image_transform_post = train_image_post_read
        self._anno_transform_post = train_label_post_read
        self._sample_len = sample_len
        self.use_tps = use_tps
        self.color_aug = color_aug
        self.nb_points = nb_points

        self._init_data()

    def _init_data(self):
        self._image_list = []
        self._image_list += list(glob.glob(os.path.join(self._root_path, "ECSSD/images/*.jpg")))
        self._image_list += list(glob.glob(os.path.join(self._root_path, "MSRA10K/images/*.jpg")))
        self._image_list += list(glob.glob(os.path.join(self._root_path, "HKU-IS/images/*.png")))


        print("Total number of data : {}".format(len(self._image_list)))


    def __len__(self):
        return(len(self._image_list))

    def _get_mask_from_image(self, image_file):
        file_name = image_file.split(".")[0].split("/images")
        file_dir = file_name[0]
        file_name = file_name[-1]

        mask_file = os.path.join(file_dir,"annotations" + file_name+".png")

        return mask_file

    def __getitem__(self, idx):
        image_file = self._image_list[idx]
        mask_file = self._get_mask_from_image(image_file)

        imageOrig = np.array(self._image_read(image_file, self._size))
        maskOrig = np.array(self._anno_read(mask_file, self._size))

        if self.use_tps:
            img, gt = aug_tps(imageOrig, maskOrig, self._sample_len, self.color_aug, self.nb_points)
        else:
            img, gt = aug_batch(imageOrig, maskOrig, self._sample_len)


        gt_list, img_list =[],[]
        for i in range(len(gt)):
            img_list.append(self._image_transform_post(Image.fromarray(img[i])))
            gt_list.append(self._anno_transform_post(Image.fromarray((gt[i]>200).astype(np.uint8))))
        images = torch.stack(img_list)


        segannos = torch.stack(gt_list)
        given_seganno = segannos[0]
        return {'images': images, 'given_seganno': given_seganno, 'segannos': segannos}

