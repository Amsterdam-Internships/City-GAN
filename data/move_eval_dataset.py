from data.base_dataset import BaseDataset, get_transform
from data.image_folder import is_image_file
from PIL import Image
import torch
import os
import glob
import random


class RoomDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.types = ["move", "real", "scanline", "random"]

        # images can be found directly in the phase folder
        self.data_dirs = [os.path.join(opt.dataroot, t) for t in self.types]

        for i in self.data_dirs:
            assert os.path.isdir(i), f"{i} is not a valid dir"

        count = 0
        self.paths = {k:[] for k in self.data_dirs}

        for directory, type_ in enumerate(self.data_dirs, self.types):
            for root, _, fnames in sorted(os.walk(directory)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        self.paths[type_].append(path)
                        count += 1
                    if count >= opt.max_dataset_size:
                        break
                if count >= opt.max_dataset_size:
                        break

        # check if the transform is the same if used multiple times (the random components)
        self.transform_img = get_transform(opt, grayscale=False)

        self.min_obj_surface = opt.min_obj_surface

        self.length = self.__len__()


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            4 different converted and transformed images (from each type 1)

        """

        out_dict = dict()

        for t in self.types:
            out_dict[t] = self.transform_img(Image.open(self.paths[t][index]).convert('RGB'))

        return out_dict


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.types) * len(self.paths[types[0]])
