import random
import glob
import os
import json
from collections import OrderedDict
import tqdm

from PIL import Image
import torch
import torchvision as tv
import numpy as np
from dataset_loaders import dataset_utils
import utils
from dataset_loaders.dataset_utils import get_sample_all, get_anno_ids, train_image_post_read, train_label_post_read
import torchvision.transforms.functional as F
import cv2
import logging
logger = logging.getLogger('global')


class YTVOSV2(torch.utils.data.Dataset):
    def __init__(self, root_path, split, image_set, impath='JPEGImages', image_read=None,
                 anno_read=None, joint_transform=None, samplelen=4, obj_selection=get_sample_all(),
                 min_num_obj=1, start_frame='random',  size=(240, 432), cacheDir="../"):
        self._min_num_objects = min_num_obj
        self._root_path = root_path
        self._split = split
        self._image_set = image_set
        self._impath = impath
        self._image_read = image_read
        self._anno_read = anno_read
        self._joint_transform = joint_transform
        self._image_transform_post = train_image_post_read
        self._anno_transform_post = train_label_post_read
        self._seqlen = samplelen
        self._obj_selection = obj_selection
        self._start_frame = start_frame
        self._size = size
        self._cacheDir = cacheDir

        self._init_data()

    def _init_data(self):
        """ Store some metadata that needs to be known during training. In order to sample, the viable sequences
        must be known. Sequences are viable if a snippet of given sample length can be selected, starting with
        an annotated frame and containing at least one more annotated frame.
        """
        print("-- YTVOS dataset initialization started.")
        if "2018" in self._root_path:
            cache_path = os.path.join(self._cacheDir, 'ytvos2018_v2_{}_100px_threshold.json'.format(self._split))
        else:
            cache_path = os.path.join(self._cacheDir, 'ytvos2019_v2_{}_100px_threshold.json'.format(self._split))


        # First find visible objects in all annotated frames
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._visible_objects, self._resolutions = json.load(f)
                self._visible_objects = {seqname: OrderedDict((int(idx), objlst) for idx, objlst in val.items())
                                         for seqname, val in self._visible_objects.items()}
            # assert len(self._visible_objects) == len(self._resolutions)
            # print("Datafile {} loaded, describing {} sequences.".format(cache_path, len(self._visible_objects)))
        else:
            # Grab all sequences in dataset
            seqnames = os.listdir(os.path.join(self._root_path, self._split, self._impath))
            list_name = os.path.join(self._root_path, "ImageSets", self._image_set + '.txt')
            with open(list_name, 'w') as filehandle:
                filehandle.writelines("%s\n" % place for place in seqnames)

            # Construct meta-info
            self._visible_objects = {}
            self._resolutions = {}
            for seqname in tqdm.tqdm(seqnames):
                anno_paths = sorted(glob.glob(self._full_anno_path(seqname, "*.png")))
                self._visible_objects[seqname] = OrderedDict(
                    (self._frame_name_to_idx(os.path.basename(path)),
                     get_anno_ids(path, dataset_utils.LabelToLongTensor(), 100))
                    for path in anno_paths)
                self._resolutions[seqname] = Image.open(anno_paths[0]).size[::-1]

            # Save meta-info
            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            with open(cache_path, 'w') as f:
                json.dump((self._visible_objects, self._resolutions), f)
            print("Datafile {} was not found, creating it with {} sequences.".format(
                cache_path, len(self._visible_objects)))

        # Find sequences in the requested image_set
        if self._split == 'train':
            with open(os.path.join(self._root_path, "ImageSets", self._image_set + '.txt'), 'r') as f:
                self._all_seqs = f.read().splitlines()
            print("{} sequences found in image set \"{}\"".format(len(self._all_seqs), self._image_set))
            # Filter out sequences that are too short from first frame with object, to last annotation
            self._nonempty_frame_ids = {
                seq: [frame_idx for frame_idx, obj_ids in lst.items() if len(obj_ids) >= self._min_num_objects]
                for seq, lst in self._visible_objects.items()}
            self._viable_seqs = [seq for seq in self._all_seqs if
                                 len(self._nonempty_frame_ids[seq]) > 0
                                 and len(self.get_image_frame_ids(seq)[min(self._nonempty_frame_ids[seq]):
                                                                       max(self._visible_objects[seq].keys()) + 1])
                                 >= self._seqlen]
            print(
                "{} sequences remaining after filtering on length (from first anno obj appearance to last anno frame.".format(
                    len(self._viable_seqs)))

            # Filter out sequences with wrong resolution
            self._viable_seqs = [seq for seq in self._viable_seqs if (self._resolutions[seq][0] >= self._size[0] and
                                                                      self._resolutions[seq][1] >= self._size[1])]
            print("{} sequences remaining after filtering out sequences that smaller than  {}.".format(
                len(self._viable_seqs), self._size))

        else:
            self._all_seqs = os.listdir(os.path.join(self._root_path, self._split, "Annotations"))
            print("{} sequences found in the Annotations directory.".format(len(self._all_seqs)))
            self._viable_seqs =  self._all_seqs
            # print(
            #     "{} sequences remaining after filtering on length (from first anno obj appearance to last anno frame.".format(
            #         len(self._viable_seqs)))

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
        return os.listdir(os.path.join(self._root_path, self._split, "Annotations", seqname))

    def get_anno_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_anno_frame_names(seqname)])

    def get_image_frame_names(self, seqname):
        return os.listdir(os.path.join(self._root_path, self._split, self._impath, seqname))

    def get_image_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])

    def get_frame_ids(self, seqname):
        """ Returns ids of all images that have idx higher than or equal to the first annotated frame"""
        all_frame_ids = sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])
        min_anno_idx = min(self.get_anno_frame_ids(seqname))
        frame_ids = [idx for idx in all_frame_ids if idx >= min_anno_idx]
        return frame_ids

    def get_nonempty_frame_ids(self, seqname):
        return self._nonempty_frame_ids[seqname]

    def _full_image_path(self, seqname, image):
        if isinstance(image, int):
            image = self._frame_idx_to_image_fname(image)
        return os.path.join(self._root_path, self._split, self._impath, seqname, image)

    def _full_anno_path(self, seqname, anno):
        if isinstance(anno, int):
            anno = self._frame_idx_to_anno_fname(anno)
        return os.path.join(self._root_path, self._split, "Annotations", seqname, anno)

    def _select_frame_ids(self, frame_ids, viable_starting_frame_ids):
        if self._start_frame == 'first':
            frame_idxidx = frame_ids.index(viable_starting_frame_ids[0])
            return frame_ids[frame_idxidx: frame_idxidx + self._seqlen]
        if self._start_frame == 'random':
            frame_idxidx = frame_ids.index(random.choice(viable_starting_frame_ids))
            return frame_ids[frame_idxidx: frame_idxidx + self._seqlen]

    def _select_object_ids(self, labels, need_select = True, this_id=0):
        if need_select :
            assert labels.min() > -1 and labels.max() < 256, "{}".format(utils.print_tensor_statistics(labels))
            possible_obj_ids = (labels[0].view(-1).bincount() > 10).nonzero().view(-1).tolist()
            if 0 in possible_obj_ids: possible_obj_ids.remove(0)
            if 255 in possible_obj_ids: possible_obj_ids.remove(255)

            obj_ids = self._obj_selection(possible_obj_ids)
        else:
            obj_ids = this_id

        bg_ids = (labels.view(-1).bincount() > 0).nonzero().view(-1).tolist()
        if 0 in bg_ids: bg_ids.remove(0)
        if 255 in bg_ids: bg_ids.remove(255)
        for idx in obj_ids:
            if idx in bg_ids: bg_ids.remove(idx)

        for idx in bg_ids:
            labels[labels == idx] = 0


        for new_idx, old_idx in zip(range(1, len(obj_ids) + 1), obj_ids):
            labels[labels == old_idx] = new_idx
        # print("Len of obj_ids : {}\t  Unique labels :{} => {}".format(obj_ids, org_unique, torch.unique(labels)))

        if need_select:
            return labels, obj_ids

        else:
            return labels

    def validity_check(self, anno, id=None):
        if id == None:
            anno=torch.from_numpy(anno).long()
            anno_idx = (anno.view(-1).bincount() > 10).nonzero().view(-1).tolist()

            # anno_idx = np.unique(anno).tolist()
            if 0 in anno_idx : anno_idx.remove(0)
            if 255 in anno_idx : anno_idx.remove(255)
            return len(anno_idx)

        else:
            is_id = np.sum((anno==id).astype(np.uint8))
            return is_id

    def __getitem__(self, idx):
        """
        returns:
            dict (Tensors): contains 'images', 'given_segmentations', 'labels'
        """
        seqname = self.get_viable_seqnames()[idx]

        # We require to begin with a nonempty frame, and will consider all objects in that frame to be tracked.
        # A starting frame is valid if it is followed by seqlen-1 frames with corresp images
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

        segannos = torch.stack(segannos)
        images = torch.stack(images)
        ##################################################################################

        try:
            segannos, this_id = self._select_object_ids(segannos, True)

        except:
            print(seqname)
            print("frame ids                ", self.get_frame_ids(seqname))
            print("frame ids post filtering ", frame_ids)
            print("viable starting frame ids", viable_starting_frame_ids)
            print("visible objects", self._visible_objects[seqname])
            raise

        segannos[segannos == 255] = 0
        given_seganno = segannos[0]

        return {'images': images, 'given_seganno': given_seganno, 'segannos': segannos}


    def _get_snippet(self, seqname, frame_ids):

        if self._split == 'valid':
            images = torch.stack(
                [(self._image_read(self._full_image_path(seqname, idx), self._size)) for idx
                 in  frame_ids]).unsqueeze(0)
            segannos = None
            anno_frame_ids = self.get_anno_frame_ids(seqname)
            given_segannos = [self._anno_read(self._full_anno_path(seqname, idx), self._size).unsqueeze(0)
                              if idx in anno_frame_ids else None for idx in frame_ids]
        else:
            images = torch.stack(
                [self._image_transform_post(self._image_read(self._full_image_path(seqname, idx), self._size)) for idx
                 in
                 frame_ids]).unsqueeze(0)
            segannos = torch.stack(
                [self._anno_transform_post(self._anno_read(self._full_anno_path(seqname, idx), self._size))
                 for idx in frame_ids]).squeeze().unsqueeze(0)
            given_segannos = [
                self._anno_transform_post(self._anno_read(self._full_anno_path(seqname, idx), self._size)).unsqueeze(0)
                if idx == self.get_anno_frame_ids(seqname)[0] else None for idx in frame_ids]
        for i in range(len(given_segannos)):  # Remove dont-care from given segannos
            if given_segannos[i] is not None:
                given_segannos[i][given_segannos[i] == 255] = 0

        fnames = [self._frame_idx_to_anno_fname(idx) for idx in frame_ids]
        return {'images': images, 'given_segannos': given_segannos, 'segannos': segannos, 'fnames': fnames}



    def _get_video(self, seqname):
        seq_frame_ids = self.get_frame_ids(seqname)
        partitioned_frame_ids = [seq_frame_ids[start_idx: start_idx + self._seqlen]
                                 for start_idx in range(0, len(seq_frame_ids), self._seqlen)]
        for frame_ids in partitioned_frame_ids:
            yield self._get_snippet(seqname, frame_ids)


    def get_video_generator(self, low=0, high=2 ** 31, test=False):

        """Returns a video generator. The video generator is used to obtain parts of a sequence. Some assumptions are made, depending on whether the train or valid splits are used. For the train split, the first annotated frame is given. No other annotation is used. For the validation split, each annotation found is given.
        """
        if test:
            for seqname in self.get_all_seqnames():
                yield (seqname, self._get_video(seqname))
        else:
            sequences = self.get_all_seqnames()[low:high]
            # NO LONGER NEEDED, now only frame ids coming after an annotated frame are utilized
            if self._split == 'train':  # These sequences are Empty in the first frame
                sequences.remove('d6917db4be')
                sequences.remove('d0c65e9e95')
                sequences.remove('c130c3fc0c')
            for seqname in sequences:
                yield (seqname, self._get_video(seqname))

