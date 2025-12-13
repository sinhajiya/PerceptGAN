import os
import random
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class PairedDataset(BaseDataset):
    """Dataset for paired folders: e.g., trainNIR/ and trainRGB/"""

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'NIR')  # e.g., /data/trainNIR
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'RGB')  # e.g., /data/trainRGB

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size] if not self.opt.serial_batches else self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('L' if self.opt.input_nc == 1 else 'RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
