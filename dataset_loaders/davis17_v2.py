import random
import glob
import os
import json
from collections import OrderedDict
from PIL import Image
import cv2
import torch
import torchvision.transforms.functional as F
from dataset_loaders import dataset_utils
import numpy as np
from dataset_loaders.dataset_utils import  get_sample_all, get_anno_ids, train_image_post_read, train_label_post_read
import torchvision as tv
import logging
logger = logging.getLogger('global')


class DAVIS17V2(torch.utils.data.Dataset):
    def __init__(self, root_path, version, image_set, image_read=None, anno_read=None, joint_transform=None, samplelen=4,
                 obj_selection=get_sample_all(), min_num_obj=1, start_frame='random', size= (480, 864), cacheDir="../",
                 ):
        self._min_num_objects = min_num_obj
        self._root_path = root_path
        self._version = version
        self._image_set = image_set
        self._image_read = image_read
        self._anno_read = anno_read
        self._image_transform_post = train_image_post_read
        self._anno_transform_post = train_label_post_read

        self._joint_transform = joint_transform
        self._seqlen = samplelen
        self._obj_selection = obj_selection
        self._start_frame = start_frame
        self._size = size
        self._cacheDir = cacheDir


        self._init_data()


    def _init_data(self):
        cache_path = os.path.join(self._cacheDir, 'davis17_v2_visible_objects_100px_threshold.json')
        # print(cache_path)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._visible_objects = json.load(f)
                self._visible_objects = {seqname: OrderedDict((int(idx), objlst) for idx, objlst in val.items())
                                         for seqname, val in self._visible_objects.items()}
        else:
            seqnames = os.listdir(os.path.join(self._root_path, 'JPEGImages', '480p'))

            self._visible_objects = {}
            for seqname in seqnames:
                anno_paths = sorted(glob.glob(self._full_anno_path(seqname, '*.png')))
                self._visible_objects[seqname] = OrderedDict(
                    (self._frame_name_to_idx(os.path.basename(path)),
                     get_anno_ids(path, dataset_utils.LabelToLongTensor(), 100))
                    for path in anno_paths)

            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            with open(cache_path, 'w') as f:
                json.dump(self._visible_objects, f)
            print("Datafile {} was not found, creating it with {} sequences.".format(cache_path,
                                                                                     len(self._visible_objects)))

        with open(os.path.join(self._root_path, 'ImageSets', self._version, self._image_set + '.txt'), 'r') as f:
            self._all_seqs = f.read().splitlines()

        self._nonempty_frame_ids = {seq: [frame_idx for frame_idx, obj_ids in lst.items() if len(obj_ids) >=
                                          self._min_num_objects]
                                    for seq, lst in self._visible_objects.items()}

        self._viable_seqs = [seq for seq in self._all_seqs if
                             len(self._nonempty_frame_ids[seq]) > 0
                             and len(self.get_image_frame_ids(seq)[min(self._nonempty_frame_ids[seq]):
                                                                   max(self._visible_objects[seq].keys()) + 1])
                             >= self._seqlen]

    def __len__(self):
        return len(self._viable_seqs)

    def _frame_idx_to_image_fname(self, idx):
        return "{:05d}.jpg".format(idx)

    def _frame_idx_to_anno_fname(self, idx):
        return "{:05d}.png".format(idx)

    def _frame_name_to_idx(self, fname):
        return int(os.path.splitext(fname)[0])

    def get_viable_seqnames(self):
        return self._viable_seqs

    def get_all_seqnames(self):
        return self._all_seqs

    def get_anno_frame_names(self, seqname):
        return os.listdir(os.path.join(self._root_path, "Annotations", "480p", seqname))

    def get_anno_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_anno_frame_names(seqname)])

    def get_image_frame_names(self, seqname):
        return os.listdir(os.path.join(self._root_path, "JPEGImages", "480p", seqname))

    def get_image_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])

    def get_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])

    def get_nonempty_frame_ids(self, seqname):
        return self._nonempty_frame_ids[seqname]

    def _full_image_path(self, seqname, image):
        if isinstance(image, int):
            image = self._frame_idx_to_image_fname(image)
        return os.path.join(self._root_path, 'JPEGImages', "480p", seqname, image)

    def _full_anno_path(self, seqname, anno):
        if isinstance(anno, int):
            anno = self._frame_idx_to_anno_fname(anno)
        return os.path.join(self._root_path, 'Annotations', "480p", seqname, anno)

    def _select_frame_ids(self, frame_ids, viable_starting_frame_ids):
        if self._start_frame == 'first':
            frame_idxidx = frame_ids.index(viable_starting_frame_ids[0])
            return frame_ids[frame_idxidx: frame_idxidx + self._seqlen]
        if self._start_frame == 'random':
            frame_idxidx = frame_ids.index(random.choice(viable_starting_frame_ids))
            return frame_ids[frame_idxidx: frame_idxidx + self._seqlen]

    def _select_object_ids(self, labels, need_select = True, this_id=0):

        if need_select :
            possible_obj_ids = (labels[0].view(-1).bincount() > 25).nonzero().view(-1).tolist()
            if 0 in possible_obj_ids:
                possible_obj_ids.remove(0)
            if 255 in possible_obj_ids:
                possible_obj_ids.remove(255)

            obj_ids = self._obj_selection(possible_obj_ids)
        else:
            obj_ids = this_id

        bg_ids = (labels.view(-1).bincount() > 0).nonzero().view(-1).tolist()
        if 0 in bg_ids:
            bg_ids.remove(0)
        if 255 in bg_ids:
            bg_ids.remove(255)
        for idx in obj_ids:
            bg_ids.remove(idx)

        for idx in bg_ids:
            labels[labels == idx] = 0
        for new_idx, old_idx in zip(range(1, len(obj_ids) + 1), obj_ids):
            labels[labels == old_idx] = new_idx
        if need_select:
            return labels, obj_ids
        else:
            return labels

    def validity_check(self, anno, id=None):
        if id == None:
            anno = torch.from_numpy(anno).long()
            anno_idx = (anno.view(-1).bincount() > 10).nonzero().view(-1).tolist()

            # anno_idx = np.unique(anno).tolist()
            if 0 in anno_idx: anno_idx.remove(0)
            if 255 in anno_idx: anno_idx.remove(255)
            return len(anno_idx)

        else:
            is_id = np.sum((anno == id).astype(np.uint8))
            return is_id


    def __getitem__(self, idx):
        seqname = self.get_viable_seqnames()[idx]

        Allframe_ids = self.get_frame_ids(seqname)
        viable_starting_frame_ids = [idx for idx in self.get_nonempty_frame_ids(seqname)
                                     if idx <= Allframe_ids[-self._seqlen]]

        frame_ids = self._select_frame_ids(Allframe_ids, viable_starting_frame_ids)
        images = [self._image_read(self._full_image_path(seqname, idx), self._size) for idx in frame_ids]
        segannos = [self._anno_read(self._full_anno_path(seqname, idx), self._size) for idx in frame_ids]

        need_hflip = random.random() < 0.5  # Do not change the need hflip during the next frames

        if self._joint_transform is not None:
            all_images = images + segannos
            images, segannos = self._joint_transform(images, segannos)
            temp = torch.from_numpy(np.array(segannos[0])).long()
            temp_list = (temp.view(-1).bincount() > 10).nonzero().view(-1).tolist()
            if 0 in temp_list: temp_list.remove(0)
            if 255 in temp_list: temp_list.remove(255)
            if len(temp_list) < 1:
                images = all_images[0:len(images)]
                segannos = all_images[len(segannos):]
                print("Wrong transformation removing translation")


        for i, (image, mask) in enumerate(zip(images, segannos)):
            if need_hflip:
                image = F.hflip(image)
                mask = F.hflip(mask)


            images[i] = self._image_transform_post(image)
            segannos[i] = self._anno_transform_post(mask)
            # print(torch.unique(segannos[i]))
        images = torch.stack(images)
        segannos = torch.stack(segannos)

        if self._version == '2017':
            segannos,this_id = self._select_object_ids(segannos, True)

        elif self._version == '2016':
            self.long = (segannos > 0).long()
            segannos = self.long

        else:
            raise ValueError("Version is not 2016 or 2017, got {}".format(self._version))

        segannos[segannos == 255] = 0
        given_seganno = segannos[0]

        return {'images': images, 'given_seganno': given_seganno, 'segannos': segannos}


    def _get_snippet(self, seqname, frame_ids):
        images = torch.stack([self._image_transform_post(self._image_read(self._full_image_path(seqname, idx), self._size))
                              for idx in frame_ids]).unsqueeze(0)
        segannos = torch.stack([self._anno_transform_post(self._anno_read(self._full_anno_path(seqname, idx), self._size))
                                for idx in frame_ids]).squeeze().unsqueeze(0)
        if self._version == '2016':
            segannos = (segannos != 0).long()
        given_segannos = [self._anno_transform_post(self._anno_read(self._full_anno_path(seqname, idx), self._size)).unsqueeze(0)
                          if idx == self.get_anno_frame_ids(seqname)[0] else None for idx in frame_ids]
        for i in range(len(given_segannos)):
            if given_segannos[i] is not None:
                given_segannos[i][given_segannos[i] == 255] = 0
                if self._version == '2016':
                    given_segannos[i] = (given_segannos[i] != 0).long()

        fnames = [self._frame_idx_to_anno_fname(idx) for idx in frame_ids]
        return {'images': images, 'given_segannos': given_segannos, 'segannos': segannos, 'fnames': fnames}


    def _get_snippetTest(self, seqname, frame_ids):
        images = torch.stack([self._image_read(self._full_image_path(seqname, idx), self._size)
                              for idx in frame_ids]).unsqueeze(0)
        segannos = torch.stack([self._anno_read(self._full_anno_path(seqname, idx), self._size)
                                for idx in frame_ids]).squeeze().unsqueeze(0)
        if self._version == '2016':
            segannos = (segannos != 0).long()
        given_segannos = [self._anno_read(self._full_anno_path(seqname, idx), self._size).unsqueeze(0)
                          if idx == self.get_anno_frame_ids(seqname)[0] else None for idx in frame_ids]


        for i in range(len(given_segannos)):
            if given_segannos[i] is not None:
                given_segannos[i][given_segannos[i] == 255] = 0
                if self._version == '2016':
                    given_segannos[i] = (given_segannos[i] != 0).long()


        fnames = [self._frame_idx_to_anno_fname(idx) for idx in frame_ids]
        return {'images': images, 'given_segannos': given_segannos, 'segannos': segannos, 'fnames': fnames}

    def _get_video(self, seqname):
        seq_frame_ids = self.get_frame_ids(seqname)
        partitioned_frame_ids = [seq_frame_ids[start_idx: start_idx + self._seqlen]
                                 for start_idx in range(0, len(seq_frame_ids), self._seqlen)]
        for frame_ids in partitioned_frame_ids:
            yield self._get_snippet(seqname, frame_ids)

    def _get_videTest(self, seqname):
        seq_frame_ids = self.get_frame_ids(seqname)
        partitioned_frame_ids = [seq_frame_ids[start_idx: start_idx + self._seqlen]
                                 for start_idx in range(0, len(seq_frame_ids), self._seqlen)]
        for frame_ids in partitioned_frame_ids:
            yield self._get_snippetTest(seqname, frame_ids)

    def get_video_generator(self, test=False):
        if test:
            for seqname in self.get_all_seqnames():
                yield (seqname, self._get_videTest(seqname))
        else:
            for seqname in self.get_all_seqnames():
                yield (seqname, self._get_video(seqname))

