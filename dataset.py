import torch
from torch.utils.data import Dataset
import os
import random
from image import *
import numpy as np
import numbers
from torchvision import datasets, transforms
import torch.nn.functional as F


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'


        fname = self.lines[index]['fname']
        img = self.lines[index]['img']
        gt_count = self.lines[index]['gt_count']

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # if random.random() > self.args['random_noise']:
            #     proportion = random.uniform(0.004, 0.015)
            #     width, height = img.size[0], img.size[1]
            #     num = int(height * width * proportion)
            #     for i in range(num):
            #         w = random.randint(0, width - 1)
            #         h = random.randint(0, height - 1)
            #         if random.randint(0, 1) == 0:
            #             img.putpixel((w, h), (0, 0, 0))
            #         else:
            #             img.putpixel((w, h), (255, 255, 255))

        gt_count = gt_count.copy()
        img = img.copy()

        if self.train == True:
            if self.transform is not None:
                img = self.transform(img)

            return fname, img, gt_count

        else:
            if self.transform is not None:
                img = self.transform(img)

            width, height = img.shape[2], img.shape[1]

            m = int(width / 384)
            n = int(height / 384)
            for i in range(0, m):
                for j in range(0, n):

                    if i == 0 and j == 0:
                        img_return = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                    else:
                        crop_img = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)

                        img_return = torch.cat([img_return, crop_img], 0).cuda()
            return fname, img_return, gt_count
