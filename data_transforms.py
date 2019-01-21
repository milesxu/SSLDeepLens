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
        return image
