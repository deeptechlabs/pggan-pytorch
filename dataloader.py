import os
import torch
import numpy as np
from io import BytesIO
import scipy.misc
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import matplotlib; matplotlib.use('agg')  # to avoid errors when display is not present
from matplotlib import pyplot as plt
from PIL import Image
import h5py


class ZeroToOneTensor(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic)
        return img.float().div(255)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImageCaptionDataset(Dataset):
    def __init__(self, images, captions, transform=None):
        self.images = images
        self.captions = captions
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        cap = self.captions[index]
        rand_caption_idx = np.random.randint(cap.shape[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, cap[rand_caption_idx]

    def __len__(self):
        assert self.images.shape[0] == self.captions.shape[0]
        return self.images.shape[0]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class dataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        self.batch_table = {4:256, 8:256, 16:256, 32:128, 64:128, 128:64, 256:64} # change this according to available gpu memory.
        self.batchsize = int(self.batch_table[pow(2,2)])        # we start from 2^2=4
        self.imsize = int(pow(2,2))
        self.num_workers = 4
        
    def renew(self, resl):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))

        self.batchsize = int(self.batch_table[pow(2,resl)])
        self.imsize = int(pow(2,resl))
        self.dataset = ImageFolder(
                    root=self.root,
                    transform=transforms.Compose(   [
                                                    transforms.Scale(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
                                                    transforms.ToTensor(),
                                                    ]))       

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)         # pixel range [-1, 1]


class hdf5_dataloader:
    def __init__(self, config):
        self.h5_file = config.train_data_h5
        self.batch_table = {4:256, 8:256, 16:256, 32:128, 64:128, 128:64, 256:64}
        self.batchsize = int(self.batch_table[pow(2,2)])
        self.imsize = int(pow(2,2))
        self.num_workers = 4

    def renew(self, resl):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.h5_file))

        self.batchsize = int(self.batch_table[pow(2,resl)])
        self.imsize = int(pow(2,resl))

        h5_data = h5py.File(self.h5_file, 'r')
        # load captions from hdf5
        captions = h5_data['/captions'][:]
        # load current resolution images from hdf5
        images = h5_data['/data{0}x{0}'.format(self.imsize)][:]
        h5_data.close()

	self.dataset = ImageCaptionDataset(images, captions,
                                           transform=ZeroToOneTensor())
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

    def get_batch(self):
        dataIter = iter(self.dataloader)
        batch_img, batch_cap = next(dataIter)
        return batch_img.mul(2).add(-1), batch_cap         # pixel range [-1, 1]
