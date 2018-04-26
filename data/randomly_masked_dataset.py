import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class RandomlyMaskedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = sorted(make_dataset(self.dir))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img = img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        # Generate random background mask.
        bg_mask = np.ones((img.height, img.width))
        num_obj = np.random.randint(1, 10)
        for _ in range(num_obj):
            w = np.random.randint(10, img.width / 4)
            h = np.random.randint(10, img.height / 4)
            x = np.random.randint(0, img.width - w)
            y = np.random.randint(0, img.height - h)
            bg_mask[y:y+h, x:x+w] = 0
        A = img
        B = Image.fromarray(np.uint8(img * np.repeat(bg_mask.reshape(bg_mask.shape[0], bg_mask.shape[1], 1), 3, 2)))

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': path, 'B_paths': path,
                'bg_mask': bg_mask}

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'RandomlyMaskedDataset'
