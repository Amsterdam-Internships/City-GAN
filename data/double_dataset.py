import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch


class DoubleDataset(BaseDataset):
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
        # get the image directory
        self.data_dir = os.path.join(opt.dataroot, opt.phase)
        # get paths to all images
        self.paths = sorted(make_dataset(self.data_dir, opt.max_dataset_size))

        self.size = self.__len__()

        self.return_mask = opt.phase == "test"

        if self.return_mask:
            self.mask_paths = [p[:-4]+"_mask"+p[-4:] for p in self.paths]

        # crop_size should be smaller than the size of loaded image
        assert(self.opt.load_size >= self.opt.crop_size)

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
        pathA = self.paths[index]
        indexB = (index + random.randint(1, self.size-1)) % self.size
        pathB = self.paths[indexB]
        # irrelevant other image
        indexC = random.randint(0, self.size-1)
        pathC = self.paths[indexC]

        A = Image.open(pathA).convert('RGB')
        B = Image.open(pathB).convert('RGB')
        C = Image.open(pathC).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)

        out = {'src': A, 'tgt': B, 'irrel': C, 'src_paths': pathA, 'tgt_paths': pathB, "irrel_paths": pathC}

        if self.return_mask:
            mask = Image.open(self.mask_paths[index]).convert('RGB')
            out['gt_num_obj'] = len(set(list(mask.getdata())))-1#len(torch.unique(mask))-1
            mask_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), method=Image.NEAREST)
            mask_visual = A_transform(mask)
            mask_og = mask_transform(mask)
            out['nearest_gt'] = mask_og
            out['visual_gt'] = mask_visual


        return out


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
