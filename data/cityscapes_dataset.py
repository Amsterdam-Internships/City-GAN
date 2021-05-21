import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

from PIL import Image
import random
import torch
from glob import glob


class CityscapesDataset(BaseDataset):
    """A dataset class for paired image dataset from Cityscapes
    A few source handpicked source images are used, and random images from the Cityscapes dataset as target images. To use, put --dataset_mode cityscapes as a command line argument.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # source directory: few handpicked files
        self.src_dir = os.path.join(opt.dataroot, "src_imgs")
        self.src_paths = sorted(make_dataset(self.src_dir, opt.max_dataset_size))
        self.src_len = len(self.src_paths)

        # get the image directory
        image_root = os.path.join(opt.dataroot, "leftImg8bit")
        self.image_dir = os.path.join(image_root, opt.phase)

        # get paths to all images
        self.tgt_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))

        self.size = self.__len__()

        self.return_mask = opt.phase == "test"

        if self.return_mask:
            # get the segmentation map directory
            segment_root = os.path.join(opt.dataroot, "gtFine")
            self.segment_dir = os.path.join(segment_root, opt.phase)
            # we want only the segmentation maps, ending in *color.png
            self.mask_paths = glob(os.path.join(self.segment_dir, "*/*_color.png"))
            # check if the number of segmentation masks equals the nr of imgs
            assert len(self.mask_paths) == len(self.tgt_paths), f"Unequal nr of images and segmentation masks ({len(self.tgt_paths)} & {len(self.mask_paths)})"

        # crop_size should be smaller/equal than the size of loaded image
        assert(self.opt.load_size >= self.opt.crop_size)

        self.input_nc = self.opt.output_nc
        self.output_nc = self.opt.input_nc



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
        # read a image given a random integer index (from the source images)
        pathA = self.src_paths[index % self.src_len]

        # get a target image
        indexB = (index)
        pathB = self.tgt_paths[indexB]

        # irrelevant other image
        indexC = random.randint(0, self.size-1)
        pathC = self.tgt_paths[indexC]

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
            # count the nr of unique pixel values to find nr of objects
            out['gt_num_obj'] = len(set(list(mask.getdata()))) - 1
            mask_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), method=Image.NEAREST)
            mask_visual = A_transform(mask)
            # no interpolation (use nearest neighbor), to prevent new pixel values
            mask_og = mask_transform(mask)
            out['nearest_gt'] = mask_og
            out['visual_gt'] = mask_visual

        return out


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.tgt_paths)
