import torch
import torch.nn.functional as F
import numpy as np


class Log10(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']
        image.log10_()
        sample['image'] = image
        return sample


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


class BoundedScale(object):
    def __init__(self, bound=1e-11, badPixel=100.0, factor0=1e11):
        self.bound = bound
        self.badPixel = badPixel
        self.factor0 = factor0

    def __call__(self, sample):

        # Load in original images.
        # I assume that the data structure of sample['image'] is a 3-D array,
        # like [image_g, image_r, image_i, image_z],
        # where image_* are 2-D arrays for presenting the images in corresponding channels.
        # Futhermore, sample['image'] stands for one system only.
        image = sample['image']

        # Remove bad pixels
        mask = image.eq(self.badPixel)
        image[mask] = 0.0
        mask = np.isnan(image)
        image[mask] = 0.0

        # Scale up images
        image = image*self.factor0

        # Clip and rescale images
        max_tensor = torch.max(image)
        image = (image - self.bound) / (max_tensor - self.bound) + self.bound
        mask = image.lt(self.bound)
        image[mask] = self.bound

        # To logarithmic scale
        image.log10_()
        mask = np.isnan(image)
        image[mask] = np.log10(self.bound)

        # Pass back processed images
        sample['image'] = image

        # If possible, I suggest that we use the fucntion
        # --- compose_class.generate_rgb_single(img_g_rscl, img_r_rscl, img_i_rscl) ---
        # to double check whether the preprocess above makes sense.
        # The usage of the function is :
        #
        #   compose_class.generate_rgb_single(sample['image'][{Band3}], \
        #                                     smaple['image'][{Band1}], \
        #                                     sample['image'][{Band2}])
        #
        # 1000 chromatic pngs should be enough.
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
