# -*- coding: utf-8 -*-

import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps


# transform for image and mask
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample["image"], sample["label"]

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size  # target size
        if w == tw and h == th:
            return {"image": img,
                    "label": mask}

        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            return {"image": img,
                    "label": mask}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {"image": img,
                "label": mask}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {"image": img,
                "label": mask}


class ChangeMask(object):

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        assert img.size == mask.size
        w, h = img.size
        new_mask = np.zeros((21, w, h))
        for i in range(w):
            for j in range(h):
                new_mask[mask[i, j]] = 1

        return {"image": img,
                "label": new_mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img,
                "label": mask}


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
        mask = np.array(sample["label"]).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img,
                "label": mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, classes=None):
        all = range(182)
        self.ignore = [x for x in all if x not in classes]
        self.classes = classes
        self.RELABELED_CLASSES_DIC = {255: 255, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 12: 11,
                                 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 21: 19, 22: 20, 23: 21, 26: 22,
                                 27: 23, 30: 24, 31: 25, 34: 26, 35: 27, 36: 28, 37: 29, 38: 30, 39: 31, 41: 32, 42: 33,
                                 43: 34, 45: 35, 46: 36, 47: 37, 48: 38, 49: 39, 50: 40, 51: 41, 52: 42, 53: 43, 54: 44,
                                 55: 45, 57: 46, 58: 47, 59: 48, 60: 49, 61: 50, 62: 51, 63: 52, 64: 53, 66: 54, 69: 55,
                                 71: 56, 72: 57, 73: 58, 74: 59, 75: 60, 76: 61, 77: 62, 78: 63, 79: 64, 80: 65, 81: 66,
                                 83: 67, 84: 68, 85: 69, 87: 70, 88: 71, 89: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77,
                                 96: 78, 97: 79, 98: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 106: 86, 107: 87,
                                 108: 88, 109: 89, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96,
                                 117: 97, 118: 98, 119: 99, 120: 100, 121: 101, 122: 102, 124: 103, 125: 104, 126: 105,
                                 127: 106, 128: 107, 129: 108, 130: 109, 131: 110, 132: 111, 133: 112, 134: 113,
                                 135: 114, 136: 115, 137: 116, 138: 117, 139: 118, 140: 119, 141: 120, 142: 121,
                                 143: 122, 145: 123, 146: 124, 149: 125, 150: 126, 151: 127, 152: 128, 153: 129,
                                 154: 130, 155: 131, 156: 132, 157: 133, 158: 134, 159: 135, 160: 136, 161: 137,
                                 162: 138, 163: 139, 164: 140, 165: 141, 166: 142, 167: 143, 169: 144, 170: 145,
                                 172: 146, 173: 147, 174: 148, 175: 149, 176: 150, 177: 151, 178: 152, 179: 153,
                                 180: 154, 181: 155, 33: 156, 40: 157, 99: 158, 56: 159, 86: 160, 32: 161, 24: 162,
                                 148: 163, 171: 164, 20: 165, 168: 166, 123: 167, 147: 168, 105: 169, 144: 170, 11: 255,
                                 25: 255, 28: 255, 29: 255, 44: 255, 65: 255, 67: 255, 68: 255, 70: 255, 82: 255,
                                 90: 255}

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample["image"]).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(sample["label"]).astype(np.float32)
        tmp = np.copy(mask)
        for i in self.ignore:
            tmp[mask == i] = 255
        for i in self.classes:
            tmp[mask == i] = self.RELABELED_CLASSES_DIC[i]

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(tmp).float()

        return {"image": img,
                "label": mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]

        assert img.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {"image": img,
                "label": mask}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # 512

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        assert img.size == mask.size
        w, h = img.size

        # if one side is 512
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {"image": img,
                    "label": mask}
        # if both sides is not equal to 512, resize to 512 * 512
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {"image": img,
                "label": mask}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        assert img.size == mask.size
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
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {"image": img,
                        "label": mask}

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
        mask = sample["label"]
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {"image": img,
                "label": mask}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        assert img.size == mask.size

        # w = int(random.uniform(0.8, 2.5) * img.size[0])
        # h = int(random.uniform(0.8, 2.5) * img.size[1])
        scale = random.uniform(0.8, 2.5)
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        sample = {"image": img, "label": mask}

        return self.crop(self.scale(sample))


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {"image": img, "label": mask}
