# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-03-09 15:00
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-05-14 16:34

"""
This script is to generate the complete dataset for evaluating the moveGAN
"""

import sys
import os

if ".." not in sys.path:
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    sys.path.append("..")

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

from util import html, evaluate, util
import numpy as np
import time


if __name__ == '__main__':

    start_time = time.time()

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    opt.batch_size = 1
    opt.num_test = 5000 if opt.data_phase=="train" else 1000

    # make sure MoveModel is loaded
    opt.epoch = "latest"
    opt.continue_train = True

    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = min(len(dataset), opt.num_test)

    # create a model given opt.model and other options
    # make sure model is loaded
    model = create_model(opt)

    # model is loaded from
    # {opt.checkpoints_dir}/{opt.name}/{opt.epoch}_net_{name}.pth
    # where opt.epoch is "latest" by default --> latest_net_G.pth
    model.setup(opt)

    # create the target directory for the Move results
    os.makedirs(f"/home/tlotze/City-GAN/datasets/ROOM_composite/{opt.data_phase}/move/run{opt.run}", exist_ok=True)

    model.eval()
    for baseline in ["move"]:
        print(f"baseline: {baseline}")
        run_folder = f"run{opt.run}/" if baseline == "move" else ""
        for i, data in enumerate(dataset):
            if not data:
                continue
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            # get output image from model
            src, composite = model.baseline(data, type_=baseline)

            # convert to numpy
            img = util.tensor2im(composite)

            # save composites
            util.save_image(img, f"/home/tlotze/City-GAN/datasets/ROOM_composite/{opt.data_phase}/{baseline}/{run_folder}{baseline}_{i}.png")





