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

        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
            '.tif', '.TIF', '.tiff', '.TIFF',
        ]

        # images can be found directly in the phase folder
        self.data_dir = os.path.join(opt.dataroot, opt.phase)

        assert os.path.isdir(self.data_dir), f"{self.data_dir} is not a valid dir"

        count = 0
        self.paths = []
        too_large=False

        for root, _, fnames in sorted(os.walk(self.data_dir)):
            for fname in fnames:
                if "img" in fname and fname.endswith('jpg'):
                    path = os.path.join(root, fname)
                    self.paths.append(path)
                    count += 1
                if count >= opt.max_dataset_size:
                    break
            if count >= opt.max_dataset_size:
                break
                too_large=True

        # check if the transform is the same if used multiple times (the random components)
        self.transform_img = get_transform(opt, grayscale=False)
        self.transform_mask = get_transform(opt, grayscale=True)

        self.min_obj_surface = opt.min_obj_surface

        self.length = self.__len__()


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            img (tensor) - - an image in one domain
            masks (tensor) - - al the masks concatenated


        This function should return:
        - tgt and source image (normal images)
        - (centered) object from source
        - corresponding centered mask from obj

        """

        out_dict = dict()

        img_path_src = self.paths[index]
        # print("src img path:", img_path_src)
        index_tgt =(index + random.randint(1, self.length-1)) % self.length
        img_path_tgt = self.paths[index_tgt]

        mask_idx = int(os.path.basename(img_path_src).split("_")[0])


        # # extract source image based on index, and random target image
        # img_path_src = os.path.join(self.data_dir, f"{index}_img.jpg")
        # index_tgt =(index + random.randint(1, self.length-1)) % self.length
        # img_path_tgt = os.path.join(self.data_dir, f"{index_tgt}_img.jpg")

        # open and convert images
        img_src = Image.open(img_path_src).convert('RGB')
        img_tgt = Image.open(img_path_tgt).convert('RGB')
        out_dict['src'] = self.transform_img(img_src)
        out_dict['tgt'] = self.transform_img(img_tgt)

        # extract all src masks, skip the floor, sky and walls
        mask_paths = sorted(glob.glob(os.path.join(self.data_dir, f"{mask_idx}_mask_*.jpg")))[4:]

        random.shuffle(mask_paths)

        # print("mask_paths:", mask_paths)

        # find suitable mask

        for p in mask_paths:
            mask = self.transform_mask(Image.open(p).convert("1"))
            mask_binary = (mask > 0).int()
            surface = mask_binary.sum().item()
            # print("surface:", surface)
            if surface > self.opt.min_obj_surface:
                out_dict["mask"] = mask
                return out_dict


        # never return None
        return self.__getitem__(random.randint(1, self.length-1))

        out_dict = None

        # # concatenate version:
        # out_dict["src_masks"] = torch.cat([self.transform_mask(Image.open(p).convert("1")) for p in mask_paths[4:]])

        # we skip the first 4 paths (floor, sky, walls)
        # for i, p in enumerate(mask_paths[4:]):
        #     mask = Image.open(p).convert("1")
        #     mask = self.transform_mask(mask)
        #     out_dict[f"src_mask{i}"] = mask

        return out_dict

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.paths)
