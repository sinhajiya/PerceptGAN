from data.base_dataset import BaseDataset, get_params, get_transform
import os
from data.image_folder import make_dataset
from PIL import Image
import random

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        if self.opt.lambda_triplet != 0:
            neg_index = random.randint(0, len(self.AB_paths) - 2)
            if neg_index >= index:
                neg_index += 1  
            neg_AB_path = self.AB_paths[neg_index]
            neg_AB = Image.open(neg_AB_path).convert('RGB')
            w_neg, h_neg = neg_AB.size
            w2_neg = int(w_neg / 2)
            B_neg_img = neg_AB.crop((w2_neg, 0, w_neg, h_neg))
            B_neg_tensor = B_transform(B_neg_img)
            B_neg_path = neg_AB_path
        
     
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path,  'B_neg': B_neg_tensor,'B_neg_paths': B_neg_path
}

        # return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'keypoints': keypoints}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class LoadData(BaseDataset):
    """
    /path/to/data/trainNIR
    /path/to/data/trainRGB
    '--dataroot /path/to/data'.
    
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'NIR') 
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'RGB')  

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
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)