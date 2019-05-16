import os
import torch
from torch.utils.data import Dataset
from astropy.table import Table
from astropy.io import fits
import numpy as np

cat = None


class GroundBasedDataset(Dataset):
    def __init__(self, root_path, offset=0, length=20000, mask_rate=0.0,
                 transform=None, use_cuda=True):
        global cat
        if cat is None:
            cat = self.load_ground_based_data(root_path)
        self.offset = offset
        self.length = length
        self.mask_rate = mask_rate
        self.labels = 2
        self.image = torch.from_numpy(
            np.array(cat['image'][offset:offset + length])).float()
        # self.data_preprocess()
        self.is_lens = torch.from_numpy(
            np.array(cat['is_lens'][offset:offset+length]))
        self.mask = torch.zeros(length, dtype=torch.uint8)
        self.make_mask()
        self.indices = torch.as_tensor(range(offset, offset+length))
        self.transform = transform
        # if batch_transform:
        #     self.image = batch_transform(self.image)
        if torch.cuda.is_available() and use_cuda:
            cuda_device = torch.device("cuda:2")
            self.image = self.image.to(cuda_device)
            self.is_lens = self.is_lens.to(cuda_device)
            self.mask = self.mask.to(cuda_device)
            self.indices = self.indices.to(cuda_device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = {'image': self.image[index],
                  'is_lens': self.is_lens[index],
                  'mask': self.mask[index],
                  'index': self.indices[index]}
        # return self.image[index], self.is_lens[index], self.mask[index], \
        #     self.indices[index]
        if self.transform:
            #sample = self.transform(sample)
            sample = self.transform(sample)
        return sample

    def load_ground_based_data(self, root_path):
        root_path = os.path.join(root_path, 'GroundBasedTraining')
        hdfile = os.path.join(root_path, 'catalogs.hdf5')
        if os.path.isfile(hdfile):
            print('loading hdf5 file...')
            return Table.read(hdfile, path='/ground')
        else:
            cat = Table.read(root_path + '/classifications.csv')
            ims = np.zeros((20000, 4, 101, 101))
            for i, id in enumerate(cat['ID']):
                for j, b in enumerate(['R', 'I', 'G', 'U']):
                    ims[i, j] = fits.getdata(
                        root_path + 'Public/Band' +
                        str(j + 1) + '/imageSDSS_' +
                        b + '-' + str(id) + '.fits'
                    )
            cat['image'] = ims
            cat.write(hdfile, path='/ground', append=True)
            return cat

    # def data_preprocess(self):
    #     vmin, vmax, scale = -1e-9, 1e-9, 100
    #     mask = self.image.eq(100)
    #     self.image[mask] = 0
    #     self.image.clamp_(vmin, vmax)
    #     self.image.div_(vmax * scale)

    def make_mask(self):
        '''
        This function will not modify self.is_lens
        So currently we do not use the label of training or not 
        '''
        mask_count = int(self.length * self.mask_rate)
        shuf_id = torch.randperm(self.length)
        count = [0, 0]
        for i in shuf_id:
            if sum(count) == mask_count:
                break
            label = int(self.is_lens[i])
            if count[label] < (mask_count // self.labels):
                self.mask[i] = 1
                count[label] += 1

        # for i in range(self.length):
        #     if not self.mask[i] > 0:
        #         self.is_lens[i] = -1.0  # unlabeled


if __name__ == "__main__":
    root_path = 'C:\\Users\\miles\\Documents\\dataset'
    dataset = GroundBasedDataset(root_path)
    print(dataset.image.shape)
    print(dataset[0]['image'].shape)
