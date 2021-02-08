from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os
import glob


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

        self.data_dir = os.path.join(opt.dataroot, opt.phase, "images_six_objects")
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc

        # check if the transform is the same if used multiple times (the random components)
        self.transform_img = get_transform(opt, grayscale=(input_nc == 1))
        self.transform_mask = get_transform(opt, grayscale=True)


        # incorporate the max length in here?
        self.length = min(len(glob.glob1(self.data_dir,"*img*")), opt.max_dataset_size)



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            img (tensor) - - an image in one domain
            masks (tensor) - - al the masks concatenated
        """

        img_path = os.path.join(self.data_dir, f"{index}_img.jpg")

        mask_paths = sorted(glob.glob(os.path.join(self.data_dir, f"{index}_mask_*.jpg")))

        img = Image.open(img_path).convert('RGB')
        img = self.transform_img(img)

        out_dict = dict()
        out_dict['img'] = img


        for i, p in enumerate(mask_paths):
            mask = Image.open(p).convert("1")
            mask = self.transform_mask(mask)
            out_dict[f"mask{i}"] = mask

        out_dict["nr_masks"] = i+1

        return out_dict

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length