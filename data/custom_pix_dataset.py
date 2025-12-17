from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os
import random
import torch


class CustomPixDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.patch_dir = os.path.join(opt.dataroot) 
        self.patch_paths = sorted(make_dataset(self.patch_dir, opt.max_dataset_size))

        assert self.opt.load_size >= self.opt.crop_size

        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc

        self.patch_len = len(self.patch_paths)
        self.AB_paths = self.patch_paths
      
    def __getitem__(self, index):

        path = self.patch_paths[index % self.patch_len]
        AB = Image.open(path).convert('RGB')
        w, h = AB.size
        w2 = w // 2
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        target_size = (self.opt.crop_size, self.opt.crop_size)
        A = A.resize(target_size, Image.BICUBIC)
        B = B.resize(target_size, Image.BICUBIC)

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

      
        return {
            'A': A,
            'B': B,
         
            'A_paths': path,
            'B_paths': path,
           
        }

    def __len__(self):
        # return max(self.patch_len, int(self.full_len / self.full_prob))
        return self.patch_len
