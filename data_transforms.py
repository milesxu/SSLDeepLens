import torch
import torch.nn.functional as F
import numpy as np


class Clamp(object):
    def __init__(self, bound, scale):
        self.bound = abs(bound)
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']
        mask = image.eq(self.scale)
        image[mask] = 0
        image.clamp_(-self.bound, self.bound)
        image.div_(self.bound * self.scale)
        sample['image'] = image
        return sample


class WhitenInput(object):
    def __init__(self, type='norm'):
        self.type = type

    def whiten_norm(self, image):
        image -= torch.mean(image, (0, 1, 2), True)
        image /= torch.mean(image ** 2, (0, 1, 2), True) ** 0.5
        return image

    def __call__(self, sample):
        image = sample['image']
        if self.type == 'norm':
            sample['image'] = self.whiten_norm(image)
        return sample


class AugmentTranslate(object):
    def __init__(self, crop_size, image_size, mirror=True):
        self.crop = crop_size
        self.i_size = image_size
        self.mirror = mirror

    def __call__(self, sample):
        image = sample['image']
        image.unsqueeze_(0)
        # print(image.size())
        p2d = tuple([self.crop] * 4)
        image = F.pad(image, p2d, 'reflect')
        image.squeeze_(0)
        # for image in images:
        if self.mirror and np.random.uniform() > 0.5:
            # image = image[:, :, ::-1]
            image = torch.flip(image, [2])
        ofs0 = np.random.randint(0, 2 * self.crop + 1)
        ofs1 = np.random.randint(0, 2 * self.crop + 1)
        image = image[:, ofs0:ofs0 + self.i_size, ofs1:ofs1 + self.i_size]
        sample['image'] = image
        return sample


class BatchPadding(object):
    def __init__(self, crop_size):
        self.crop = crop_size

    def __call__(self, images):
        p2d = tuple([self.crop] * 4)
        return F.pad(images, p2d, 'reflect')
