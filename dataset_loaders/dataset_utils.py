import os
import sys
import numpy as np
import torch
import torch.utils.data
import random
from torch.utils.data import Sampler
import torchvision as tv
from PIL import Image
import math
import torch.nn.functional as F
import numbers


def get_sample_bernoulli(p):
    return (lambda lst: [elem for elem in lst if random.random() < p])
def get_sample_all():
    return (lambda lst: lst)
def get_sample_k_random(k):
    return (lambda lst: sorted(random.sample(lst, min(k,len(lst)))))

def get_anno_ids(anno_path, pic_to_tensor_function, threshold):
    pic = Image.open(anno_path)
    tensor = pic_to_tensor_function(pic)
    values = (tensor.view(-1).bincount() > threshold).nonzero().view(-1).tolist()
    if 0 in values: values.remove(0)
    if 255 in values: values.remove(255)
    return values

def get_255anno_ids(anno_path, pic_to_tensor_function, threshold):
    pic = Image.open(anno_path)
    tensor = pic_to_tensor_function(pic)/255
    values = (tensor.view(-1).bincount() > threshold).nonzero().view(-1).tolist()

    if 0 in values: values.remove(0)
    if 255 in values: values.remove(255)
    return values

IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD = [.229, .224, .225]


class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            label = torch.from_numpy(pic).long()
        elif pic.mode == '1':
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            if pic.mode == 'LA':
                label = label.view(pic.size[1], pic.size[0], 2)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
                label = label.view(1, label.size(0), label.size(1))
            else:
                label = label.view(pic.size[1], pic.size[0], -1)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
        return label
class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, labels):
        for t in self.transforms:
            images, labels = t(images, labels)
        return images, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
            format_string += '\n)'
        return format_string

class JointRandomHorizontalFlip(object):
    def __call__(self, *args):
        if random.choice([True, False]):
            out = []
            for tensor in args:
                idx = [i for i in range(tensor.size(-1)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                tensor_flip = tensor.index_select(-1, idx)
                out.append(tensor_flip)
            return out
        else:
            return args

def centercrop(tensor, cropsize):
    _, _, H, W = tensor.size()
    A, B = cropsize
    return tensor[:, :, (H-A)//2:(H+A)//2, (W-B)//2:(W+B)//2]


def random_object_sampler(lst):
    return [random.choice(lst)]


def deterministic_object_sampler(lst):
    return [lst[0]]

def train_image_init_read(path, size = (320, 320)):
    pic = Image.open(path)
    if pic.mode != 'RGB':
        pic=pic.convert('RGB')
    transform = tv.transforms.Compose(
        [tv.transforms.RandomGrayscale(0.1),
         tv.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
         tv.transforms.Resize(size, interpolation=Image.BILINEAR)])
    return transform(pic)


def train_label_init_read(path, size = (320, 320)):
    if os.path.exists(path):
        #pic = np.load(path).astype(np.uint8)
        pic = Image.open(path)

        transform = tv.transforms.Compose(
            [tv.transforms.Resize(size, interpolation=Image.NEAREST)])
        label = transform(pic)
    else:
        label = F.to_pil_image(torch.LongTensor(1,*size).fill_(255)) # Put label that will be ignored
    return label


def train_image_post_read(pic, size = (480, 480)):
    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform(pic)


def train_label_post_read(pic, size = (480, 480)):
    transform = tv.transforms.Compose([LabelToLongTensor()])
    label = transform(pic)
    return(label)


def train_image_read(path, size):
    pic = Image.open(path)
    transform = tv.transforms.Compose(
        [tv.transforms.RandomGrayscale(0.2),
         tv.transforms.Resize(size, interpolation=Image.BILINEAR),
         tv.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
         tv.transforms.ToTensor(),
         tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform(pic)


def train_augmentation_image_read(path, size):
    pic = Image.open(path)
    transform = tv.transforms.Compose(
        [
            tv.transforms.RandomAffine(30, scale=(0.75, 1.25), shear=30),
            tv.transforms.Resize(size, interpolation=Image.BILINEAR),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform(pic)


def train_label_read(path, size):
    if os.path.exists(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.Resize(size, interpolation=Image.NEAREST),
             LabelToLongTensor()])
        label = transform(pic)
    else:
        label = torch.LongTensor(1, *size).fill_(255)  # Put label that will be ignored
    return label

class EmptyTransform():
    def __call__(self, image, mask):
        return (image, mask)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, d1, d2, batch_size):
        self.d1 = d1
        self.d2 = d2
        self.batch_size = batch_size

        self.l1 = len(d1)
        self.l2 = len(d2)
        # print ("L1 : " + str(self.l1) + " - L2 : " + str(self.l2))
        self.split = math.ceil(self.l1 / self.batch_size)
        # print ('s = ' + str(self.split))
        self.split = self.split * self.batch_size
        # print ("size : " + str(self.split))
        self.idx1 = np.arange(self.l1)
        np.random.shuffle(self.idx1)
        self.idx2 = np.arange(self.l2)
        np.random.shuffle(self.idx2)

    def __getitem__(self, i):
        if i < self.split:
            return self.d1[self.idx1[i % self.l1]]
        else:
            return self.d2[self.idx2[(i - self.split) % self.l2]]

    def __len__(self):
        return self.split + self.l2


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def test_imagelist_transform(pic_list):
    new_list = []
    transform = tv.transforms.Compose(
        [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    for pic in pic_list:
        new_list.append(transform(pic))
    return new_list



class MyBatchSampler(Sampler):

    def __init__(self, batch_size, l1, l2):
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self.generateIdxs()

    def generateIdxs(self):
        s1 = math.ceil(self.l1 / self.batch_size)

        idx1 = np.arange(self.l1)
        np.random.shuffle(idx1)
        idx1 = list(chunks(list(idx1), self.batch_size))
        idx2 = np.arange(s1 * self.batch_size, s1 * self.batch_size + self.l2)
        np.random.shuffle(idx2)
        idx2 = list(chunks(list(idx2), self.batch_size))

        idxs = idx1 + idx2
        np.random.shuffle(idxs)
        # pprint.pprint(idxs)
        self.idxs = idxs

    def __iter__(self):
        self.generateIdxs()  # regenerate the idxs at each epoch

        for v in self.idxs:
            yield list(v)

    def __len__(self):
        return (self.l1 // self.batch_size) + (self.l2 // self.batch_size)



_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class JointImageVidTransform(object):
    def __init__(self, degrees, crop=0, hflip =0, vflip=0, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if isinstance(crop, numbers.Number):
            self.crop_size = (crop, crop)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "crop should be a list or tuple and it must be of length 2."
            self.crop_size = crop

        if hflip is not None:
            if not (0.0 <= hflip <= 1.0):
                raise ValueError("horizontal flip prob should be between 0 and 1")
        self.hflip = hflip

        if vflip is not None:
            if not (0.0 <= vflip <= 1.0):
                raise ValueError("vertical flip prob should be between 0 and 1")
        self.vflip = vflip

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params_crop(img_size, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, imgs, labels):
        out = []
        if (self.crop_size[0] > 0) and ((self.crop_size[0] > imgs[0].size[0]) or (self.crop_size[1] > imgs[0].size[1])):
            out_crop = []
            left_pad = max(0, self.crop_size[1] - imgs[0].size[1])
            top_pad = max(0, self.crop_size[0] - imgs[0].size[0])
            for curr_image in imgs:
                out_crop.append(F.pad(curr_image, (left_pad, top_pad, 0, 0)))
            imgs = out_crop
        if (self.crop_size[0] > 0):
            out_crop = []
            i, j, h, w = self.get_params_crop(imgs[0].size, self.crop_size)
            for curr_image in imgs:
                out_crop.append(F.crop(curr_image, i, j, h, w))
            imgs = out_crop

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, imgs[0].size)
        for img_id in range(len(imgs)):
            out.append(tv.transforms.functional.affine(imgs[img_id], *ret, resample=Image.BILINEAR, fillcolor=self.fillcolor))
        for img_id in range(len(labels)):
            out.append(tv.transforms.functional.affine(labels[img_id], *ret, resample=Image.NEAREST, fillcolor=self.fillcolor))

        if random.random() < self.hflip:
            for idx in range(len(out)):
                out[idx] = F.hflip(out[idx])

        if random.random() < self.vflip:
            for idx in range(len(out)):
                out[idx] = F.vflip(out[idx])
        img_num = len(out) //2

        return out[:img_num], out[img_num:]

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)
#end