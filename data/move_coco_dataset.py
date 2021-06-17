from data.base_dataset import BaseDataset, get_transform
from data.image_folder import is_image_file, make_dataset
from collections import defaultdict
from PIL import Image, ImageDraw
import torch
import os
import glob
import random
import json
import numpy as np


class MoveCocoDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    This model loads source images from the COCO dataset, and target images from the Cityscapes dataset, for the MoveGAN. It picks an object from the COCO data to be inserted into the target image

    To use, use --dataset_mode move_coco

    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be
             a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # images can be found directly in the phase folder
        self.src_dir = os.path.join(opt.dataroot, "src_imgs/images")
        self.src_paths = sorted(make_dataset(self.src_dir,
            opt.max_dataset_size))
        self.src_ids = [os.path.basename(fname)[:-4].split("_")[1] for fname in self.src_paths]
        self.id2path_src = {id_ : path for id_, path in zip(self.src_ids, self.src_paths)}
        self.src_len = len(self.src_ids)

        # get transforms for masks and images
        self.transform_img = get_transform(opt, grayscale=False)
        self.transform_mask = get_transform(opt, grayscale=True)
        
        with open(os.path.join(opt.dataroot, "src_imgs/annotations",
            "COCO_anns_complete.json")) as f:
            self.COCO_anns = json.load(f)
            self.anns = self.COCO_anns['img_anns']
            self.categories = self.COCO_anns['cats']
            self.polygon_dict = self.create_polygon_masks(self.anns)
            # print(self.polygon_dict)

        # get the image directory for Cityscapes (target)
        image_root = os.path.join(opt.dataroot, "leftImg8bit")
        self.image_dir = os.path.join(image_root, opt.phase)
        assert os.path.isdir(self.image_dir), f"{self.image_dir} is \
            not a valid dir"

        # get paths to all cityscapes images
        self.tgt_paths = sorted(make_dataset(self.image_dir,
            opt.max_dataset_size))

        self.min_obj_surface = opt.min_obj_surface

        self.length = self.__len__()

    def create_polygon_masks(self, anns_dict):
        """This function takes a dictionary with all the annotations
        for all objects in the source image, which contains the polygon coordinates. A dictionary is constructed with all the polygon masks"""
        polygon_dict = defaultdict(list)

        for img_id, ann in anns_dict.items():
            print("img_id", img_id)
            surfaces, masks = [], []
            w, h = Image.open(self.id2path_src[img_id]).convert('RGB').size
            for obj in ann:
                img = Image.new('1', (w, h), 0)
                seg = obj['segmentation']
                if type(seg) != list:
                    continue
                polygon = np.array(seg[0]).reshape((int(len(seg[0])/2), 2))
                polygon = [(x, y) for x, y in polygon]
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                mask = self.transform_mask(img)
                mask_binary = (mask > 0).int()
                # surface = mask_binary.sum().item()
                # if surface > self.opt.min_obj_surface:
                    # surfaces.append(surface)
                masks.append(mask)

            # we could sort the polygon_dict[img_id] here on surface size
            # sorted_masks = [x for _, x in sorted(zip(surfaces, masks), key = lambda x: x[0], reverse=True)]
            polygon_dict[img_id] = masks

        return polygon_dict


    def get_random_object_mask(self, src_index):
        src_id = self.src_ids[src_index]
        return random.choice(self.polygon_dict[src_id])



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

        # Extract images from COCO dataset
        src_index = index % self.src_len
        img_path_src = self.src_paths[src_index]
        img_id_src = self.src_ids[src_index]

        # extract target images from cityscapes
        img_path_tgt = self.tgt_paths[index]

        # open and convert images
        img_src = Image.open(img_path_src).convert('RGB')
        img_tgt = Image.open(img_path_tgt).convert('RGB')
        out_dict['src'] = self.transform_img(img_src)
        out_dict['tgt'] = self.transform_img(img_tgt)
        out_dict['mask'] = self.get_random_object_mask(src_index)

        return out_dict

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.tgt_paths)
