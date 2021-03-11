# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-03-09 15:00
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-03-11 13:33

"""
This script is for testing any model. It is similar to train.py in setup, but evaluates on a test set without updating the model. It loads the model from memory instead.
"""


import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, evaluate


# for testing purposes
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


def split_mask(mask):
    """
    extract the number of different values for
    the b_channel (3rd channel)

    assume mask is a 3 channel tensor

    the output is a 4D tensor, with each mask over the batch dimension

    """

    assert mask.shape[0] == 3
    dim = mask.shape[-1]

    # extract the blue channel
    b_channel = mask[2, :, : ]
    # find the unique values in the tensor, these correspond to the masks
    values = torch.unique(b_channel)
    # initialize the 4D output tensor
    out = torch.zeros(values.size()[0], 1, dim, dim)

    for i, v in enumerate(values):
        # skip background
        if v == 0:
            continue
        # extract the object mask from blue channel
        binary_mask = torch.where(b_channel == v, 1, 0)
        # set in output batch
        out[i, 0, :, :] = binary_mask

    return out




if __name__ == '__main__':


    mask = transforms.ToTensor()(Image.open('datasets/CLEVR_colorized/images/test/CLEVR_color__028002_mask.png').convert('RGB'))

    mask_split = split_mask(mask)



    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.


    ### temporary
    opt.batch_size = 1
    opt.display_freq = 1



    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options



    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))


    IOU_list = []

    model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.test(data)           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = str(i)

        # compute IOU
        batch_iou = evaluate.compute_IOU(model.g_mask_binary, model.bin_gt)
        IOU_list.extend(batch_iou)


        if i % opt.display_freq == 0:
            iou= f"{batch_iou[0]:.2f}"
            # print(f"IOU {i}: {iou}")
            # print(f'processing and saving {i}-th image... {img_path}')
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, score=iou)
    webpage.save()  # save the HTML

    # print IOU information
    mean, min_, max_, std = np.mean(IOU_list), np.min(IOU_list), np.max(IOU_list), np.std(IOU_list)

    print(f"\nMean: {mean:.2f}\nMin: {min_:.2f}\nMax: {max_:.2f}\nSTD: {std:.2f}")









