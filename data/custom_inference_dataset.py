import os
import torch
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

class CustomInferenceDataset(BaseDataset):
  
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        self.patch_size = opt.crop_size        # 128
        self.stride = 64
        self.image_size = opt.load_size        # 256

        self.transform = get_transform(
            opt, 
            params=None, 
            grayscale=(opt.input_nc == 1)
        )

        # Precompute patch grid
        self.grid = []
        for y in range(0, self.image_size - self.patch_size + 1, self.stride):
            for x in range(0, self.image_size - self.patch_size + 1, self.stride):
                self.grid.append((x, y))

        self.num_patches = len(self.grid)

    def __len__(self):
        return len(self.paths) * self.num_patches

    def __getitem__(self, index):
        img_idx = index // self.num_patches
        patch_idx = index % self.num_patches
        x, y = self.grid[patch_idx]

        path = self.paths[img_idx]
        AB = Image.open(path).convert('RGB')

        w, h = AB.size
        w2 = w // 2

        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        A = A.crop((x, y, x + self.patch_size, y + self.patch_size))
        B = B.crop((x, y, x + self.patch_size, y + self.patch_size))

        A = self.transform(A)
        B = self.transform(B)

        return {
            'A': A,
            'B': B,
            'x': x,
            'y': y,
            'img_id': os.path.basename(path),
            'A_paths': path,
            'B_paths': path
        }