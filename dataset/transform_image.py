# -*- coding: utf-8 -*-

import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps
# transform for image and label
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, label = sample["image"], sample["label"]

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {"image": img,
                    "label": label}

        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            return {"image": img,
                    "label": label}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))

        return {"image": img,
                "label": label}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))

        return {"image": img,
                "label": label}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img,
                "label": label}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample["image"]).astype(np.float32)
        label = np.array(sample["label"]).astype(np.int32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img,
                "label": label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample["image"]).astype(np.float32).transpose((2, 0, 1))
        label = np.array(sample["label"])

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).int()

        return {"image": img,
                "label": label}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]

        img = img.resize(self.size, Image.BILINEAR)

        return {"image": img,
                "label": label}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size#512

    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]
        w, h = img.size

        # if one side is 512
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {"image": img,
                    "label": label}
        # if both sides is not equal to 512, resize to 512 * 512
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)

        return {"image": img,
                "label": label}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)

                return {"image": img,
                        "label": label}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)

        return {"image": img,
                "label": label}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]

        #w = int(random.uniform(0.8, 2.5) * img.size[0])
        #h = int(random.uniform(0.8, 2.5) * img.size[1])
        scale = random.uniform(0.8, 2.5)
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img= img.resize((w, h), Image.BILINEAR)
        sample = {"image": img, "label": label}

        return self.crop(self.scale(sample))

class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample["image"]
        label = sample["label"]

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img = img.resize((w, h), Image.BILINEAR)

        return {"image": img, "label": label}