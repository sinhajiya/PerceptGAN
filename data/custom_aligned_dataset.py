# data/custom_dataset.py
from torch.utils.data import DataLoader
from data.base_dataset import BaseDataset  # required by framework
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torch

class CustomAlignedDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        if not self.opt.dontaugment:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'NIR')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'RGB')
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + '_NIR')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '_RGB')

        self.A_paths = sorted(self._make_dataset(self.dir_A))
        self.B_paths = sorted(self._make_dataset(self.dir_B))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        
        
        self.transform_A = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_B = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        # Load and convert images
        A_img = Image.open(A_path).convert('L')
        B_img = Image.open(B_path).convert('RGB')


        # Resize to the same size if needed
        if A_img.size != B_img.size:
            target_size = (min(A_img.size[1], B_img.size[1]), min(A_img.size[0], B_img.size[0]))
            A_img = A_img.resize(target_size[::-1], Image.BICUBIC)
            B_img = B_img.resize(target_size[::-1], Image.BICUBIC)


        # if self.opt.phase == 'train':
        if self.opt.phase == 'train':
            width, height = A_img.size
            if width > self.load_size and height > self.load_size:
                x = (width - self.load_size) // 2
                y = (height - self.load_size) // 2
                A_img = A_img.crop((x, y, x + self.load_size, y + self.load_size))
                B_img = B_img.crop((x, y, x + self.load_size, y + self.load_size))
                
            else:
                A_img = A_img.resize((self.load_size, self.load_size), Image.BICUBIC)
                B_img = B_img.resize((self.load_size, self.load_size), Image.BICUBIC)
                
        # ---- 50% chance of single random augmentation ----
            if not self.opt.dontaugment:
                if random.random() < 0.5:
                    choices = ['flip', 'sharpness','crop']
                
                    aug_type = random.choice(choices)

                    if aug_type == 'crop':
                        # Random zoom-in crop, then resize back
                        scale = random.uniform(0.8, 1.0)
                        new_crop = int(self.crop_size * scale)
                        x = random.randint(0, self.load_size - new_crop)
                        y = random.randint(0, self.load_size - new_crop)
                        # A_img = A_img.crop((x, y, x + new_crop, y + new_crop))
                        # B_img = B_img.crop((x, y, x + new_crop, y + new_crop))
                        
                        A_img = A_img.crop((x, y, x + new_crop, y + new_crop)).resize((self.crop_size, self.crop_size), Image.BICUBIC)
                        B_img = B_img.crop((x, y, x + new_crop, y + new_crop)).resize((self.crop_size, self.crop_size), Image.BICUBIC)
                        

                    elif aug_type == 'flip':
                        A_img = A_img.transpose(Image.FLIP_LEFT_RIGHT)
                        B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
                       

                    elif aug_type == 'sharpness':
                        sharp_factor = random.uniform(0.5, 2.0)
                        A_img = transforms.functional.adjust_sharpness(A_img, sharp_factor)
                        B_img = transforms.functional.adjust_sharpness(B_img, sharp_factor)
                       
                else:
                    x = (self.load_size - self.crop_size) // 2
                    y = (self.load_size - self.crop_size) // 2
                    A_img = A_img.crop((x, y, x + self.crop_size, y + self.crop_size))
                    B_img = B_img.crop((x, y, x + self.crop_size, y + self.crop_size))
                   
            else:
                x = (self.load_size - self.crop_size) // 2
                y = (self.load_size - self.crop_size) // 2
                A_img = A_img.crop((x, y, x + self.crop_size, y + self.crop_size))
                B_img = B_img.crop((x, y, x + self.crop_size, y + self.crop_size))
               
            A_img = A_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            B_img = B_img.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        A_tensor = self.transform_A(A_img)
        B_tensor = self.transform_B(B_img)
        
        return {
            'A': A_tensor,
            'B': B_tensor,
           
            'A_paths': A_path,
            'B_paths': B_path,
            
        }
    def __len__(self):
        return max(self.A_size, self.B_size)

    def _make_dataset(self, dir):
        return [os.path.join(dir, fname) for fname in sorted(os.listdir(dir))
                if any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
