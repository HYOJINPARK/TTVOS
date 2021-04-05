import torch
import torch.utils.data as data
import torch.nn as nn

import random
from os.path import join
import numpy as np

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from .tps import tps_grid_to_remap, tps_theta_from_points, tps_grid

def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.0))
    j = (j+1)/2
    return int(j)

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.Upsample(size=(size, size), mode='bilinear')

    labelVar = torch.from_numpy(label.transpose(3, 2, 0, 1))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>0.3]  = 1
    label_resized[label_resized != 0]  = 1
    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def aug_batch(img, gt, nb_frames):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes2 = lambda aug: iaa.Sometimes(0.9, aug)

    seq = iaa.Sequential(
        [
            sometimes2(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.Add((-10, 10), per_channel=0.5),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ], random_order=True
        )
    
    seq2 = iaa.Sequential(
        [
            sometimes2(iaa.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-10, 10), # shear by -16 to +16 degrees
                order=0, # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            #sometimes2(iaa.CoarseDropout(0.2, size_percent=(0.1, 0.5)
            #))
        ], random_order=True
        )

    images = [img for i in range(0, nb_frames)]
    segmaps = [gt for i in range(0, nb_frames)]

    images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)

    return (images_aug, segmaps_aug)

    scale = random.uniform(0.5, 1.3)
    #random.uniform(0.5,1.5) does not fit in a Titan X with the present version of pytorch,
    # so we random scaling in the range (0.5,1.3), different than caffe implementation
    # in that caffe used only 4 fixed scales. Refer to read me
    scale=1
    dim = int(scale*321)

    flip_p = random.uniform(0, 1)

    img_temp = flip(img,flip_p)
    gt_temp = flip(gt,flip_p)

    seq_det = seq.to_deterministic()
    img_temp = seq_det.augment_image(img_temp)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(float)/255.
    img_temp = cv2.resize(img_temp,(dim,dim))

    gt_temp = ia.SegmentationMapOnImage(gt_temp, shape=gt_temp.shape, nb_classes=2)
    gt_temp_map = seq_det.augment_segmentation_maps([gt_temp])[0]
    gt_temp = gt_temp_map.get_arr_int().astype(float)
    mask = seq2.augment_segmentation_maps([gt_temp_map])[0].get_arr_int()
    mask = cv2.resize(mask,(dim,dim) , interpolation = cv2.INTER_NEAREST).astype(float)


    return (img_temp, mask)

    kernel = np.ones((int(scale*5), int(scale*5)), np.uint8)
    
    bb = cv2.boundingRect(gt_temp.astype('uint8'))
 
    if bb[2] != 0 and bb[3] != 0:
        fc = np.ones([dim, dim, 1]) * -100/255.
        #fc[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2], 0] = 100
        if flip_p <= 1.0:
            aug_p = random.uniform(0, 1)
            it = random.randint(1, 5)

            aug = np.expand_dims(cv2.dilate(mask, kernel, iterations=it), 2)
            fc[np.where(aug==1)] = 100/255.
    else:
        fc = np.ones([dim, dim, 1]) * -100/255.
    
    image = np.dstack([img_temp, fc])
    gt_temp = np.expand_dims(gt_temp, 2)
    gt = np.expand_dims(gt_temp, 3)
    label = resize_label_batch(gt, outS(dim))
    label = label.squeeze(3)

    return image, label

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def wrap_random_cv(img, gt, nb_points = 5):

    bb = cv2.boundingRect(gt.astype('uint8'))
    #print ('bb : ', bb, gt.shape)


    h = bb[3]
    w = bb[2]
    xs = np.random.randint(bb[0], bb[0] + w, size=nb_points)
    ys = np.random.randint(bb[1], bb[1] + h, size=nb_points)

    xs_dst = []
    ys_dst = []
    for l in range(0, nb_points):
        shift = np.random.uniform(-0.1, 0.1, 2)
        xs_dst.append(xs[l] + shift[0] * w)
        ys_dst.append(ys[l] + shift[1] * h)

    # divide the x and y to be betweend 0 and 1
    img_height = gt.shape[0]
    img_width = gt.shape[1]

    xs = np.array(xs) / img_width
    ys = np.array(ys) / img_height
    xs_dst = np.array(xs_dst) / img_width
    ys_dst = np.array(ys_dst) / img_height
    #print ("xs", xs, "ys", ys)
    #print ("xs_dst", xs_dst, "ys", ys_dst)
    # Clip between 0 and 1 the xs and ys coordinate
    xs2 = np.clip(xs, 0, 1)
    ys2 = np.clip(ys, 0, 1)
    xs_dst2 = np.clip(xs_dst, 0, 1)
    ys_dst2 = np.clip(ys_dst, 0, 1)

    c_src = np.stack((xs2, ys2), axis=-1)
    c_dst = np.stack((xs_dst2, ys_dst2), axis=-1)
    #print ("c_src", c_src, "c_dst", c_dst)

    try:
        warped_img = warp_image_cv(img, c_src, c_dst)
        warped_gt = warp_image_cv(gt, c_src, c_dst)
    except Exception as e:
        print ("Error", e)
        print ("xs", xs, "ys", ys)
        print ("xs_dst", xs_dst, "ys", ys_dst)
        print ("c_src", c_src, "c_dst", c_dst)
        return (img, gt)

    return (warped_img, warped_gt)

def aug_tps(img, gt, nb_frames, color_aug = False, nb_points = 5):

    seq = iaa.Sequential([
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ], random_order=True) 

    #bb = cv2.boundingRect(gt.astype('uint8'))
    #print ('bb : ', bb, gt.shape)

    images = [img]
    segmaps = [gt]
    for l in range(1, nb_frames):
        i, m = wrap_random_cv(img, gt, nb_points)
        if color_aug:
            m2 = np.expand_dims(m, axis=2)
            i_aug, m_aug = seq(image=i, segmentation_maps=[m2])

            images.append(i_aug)
            segmaps.append(m_aug[0])
        else:
            images.append(i)
            segmaps.append(m)
        

    return (images, segmaps)
