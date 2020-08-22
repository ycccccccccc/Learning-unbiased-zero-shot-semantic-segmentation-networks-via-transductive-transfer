# -*- coding: utf-8 -*-
import sys

import os
import shutil
from PIL import Image
from dataset.transform_pixel_coco import *
from dataset.util import *
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

def transform_for_train(fixed_scale=512, rotate_prob=15, classes=None):
    """
    Options:
    1.RandomCrop
    2.CenterCrop
    3.RandomHorizontalFlip
    4.Normalize
    5.ToTensor
    6.FixedResize
    7.RandomRotate
    """
    transform_list = []
    # transform_list.append(FixedResize(size = (fixed_scale, fixed_scale)))
    transform_list.append(RandomSized(fixed_scale))
    transform_list.append(RandomRotate(rotate_prob))
    transform_list.append(RandomHorizontalFlip())
    transform_list.append(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform_list.append(ToTensor(classes = classes))

    return transforms.Compose(transform_list)


def transform_for_test(fixed_scale=512, classes=None):
    transform_list = []
    # transform_list.append(FixedResize(fixed_scale))
    transform_list.append(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform_list.append(ToTensor(classes = classes))
    return transforms.Compose(transform_list)


def prepare_for_train_pixel_dataloader(dataroot, bs_train=4, input_size=(512, 512), shuffle=True, num_workers=2):
    """
    use pascal 2012, dataroot contain JEPG/SEGMENTATION/so on
    """
    with open(os.path.join(dataroot, "train_strong.txt"), "r") as f:
        lines = f.read().splitlines()
        classes = lines[0]
        lines.remove(classes)
        classes = classes.split(",")
        classes = list(map(int, classes))

    transform = transform_for_train(fixed_scale=input_size, rotate_prob=15, classes=classes)

    voc_train = VOCSegmentation(base_dir=dataroot, split="train_strong", transform=transform)

    dataloader = DataLoader(voc_train, batch_size=bs_train, shuffle=shuffle, num_workers=num_workers,
                            drop_last=True)

    return dataloader

def prepare_for_train_weak_pixel_dataloader(dataroot, bs_train=4, input_size=(512, 512), shuffle=True, num_workers=2):
    """
    use pascal 2012, dataroot contain JEPG/SEGMENTATION/so on
    """
    with open(os.path.join(dataroot, "train_weak.txt"), "r") as f:
        lines = f.read().splitlines()
        classes = lines[0]
        lines.remove(classes)
        classes = classes.split(",")
        classes = list(map(int, classes))

    transform = transform_for_train(fixed_scale=input_size, rotate_prob=15, classes=classes)

    voc_train = VOCSegmentation(base_dir=dataroot, split="train_weak", transform=transform)

    dataloader = DataLoader(voc_train, batch_size=bs_train, shuffle=shuffle, num_workers=num_workers,
                            drop_last=True)

    return dataloader

def prepare_for_val_pixel_dataloader(dataroot, bs_val=6, input_size=512, shuffle=False, num_workers=2):
    """
    use pascal 2012
    """
    with open(os.path.join(dataroot, "val.txt"), "r") as f:
        lines = f.read().splitlines()
        classes = lines[0]
        lines.remove(classes)
        classes = classes.split(",")
        classes = list(map(int, classes))

    transform = transform_for_test(fixed_scale=input_size,classes = classes)

    voc_test = VOCSegmentation(base_dir=dataroot, split="val", transform=transform)

    dataloader = DataLoader(voc_test, batch_size=bs_val, shuffle=shuffle, num_workers=num_workers)

    return dataloader


def prepare_for_val_weak_pixel_dataloader(dataroot, bs_val=6, input_size=512, shuffle=False, num_workers=2):
    """
    use pascal 2012
    """
    with open(os.path.join(dataroot, "val_weak.txt"), "r") as f:
        lines = f.read().splitlines()
        classes = lines[0]
        lines.remove(classes)
        classes = classes.split(",")
        classes = list(map(int, classes))

    transform = transform_for_test(fixed_scale=input_size,classes = classes)

    voc_test = VOCSegmentation(base_dir=dataroot, split="val_weak", transform=transform)

    dataloader = DataLoader(voc_test, batch_size=bs_val, shuffle=shuffle, num_workers=num_workers)

    return dataloader


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir="./dataset",
                 split="train",
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, "JPEGImages")
        self._cat_dir = os.path.join(self._base_dir, "SegmentationClassAug")

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(_splits_dir, splt + ".txt"), "r") as f:
                lines = f.read().splitlines()
                classes = lines[0]
                lines.remove(classes)

            for ii, line in enumerate(lines):
                _image_name, _cat_name = line.split(" ")
                _image_name = _image_name[1:]
                _cat_name = _cat_name[1:]
                _image = os.path.join(self._base_dir, _image_name)
                _cat = os.path.join(self._base_dir, _cat_name)
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print("Number of images in {}: {:d}".format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {"image": _img, "label": _target}
        _name = self.categories[index].split("/")[-1]
        _size = _img.size

        if self.transform is not None:
            sample = self.transform(sample)
        sample["name"] = _name
        sample["size"] = str(_size[0]) + "," + str(_size[1])
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert("RGB")).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return "VOC2012(split=" + str(self.split) + ")"
