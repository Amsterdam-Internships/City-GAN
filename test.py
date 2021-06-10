# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-03-09 15:00
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-06-10 13:26

"""
This script is for testing any model. It is similar to train.py in setup, but evaluates on a test set without updating the model. Instead, the model is loaded from memory.
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, evaluate
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
    if opt.batch_size != 1 and opt.model != "classifier":
        opt.batch_size = 1
        print("Batch size is set to 1 for testing")

    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset_size = min(len(dataset), opt.num_test)
    # create a model given opt.model and other options
    model = create_model(opt)

    # model is loaded from
    # {opt.checkpoints_dir}/{opt.name}/{opt.epoch}_net_{name}.pth
    # where opt.epoch is "latest" by default --> latest_net_G.pth
    model.setup(opt)

    # create the webpage
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    if opt.model != "classifier":
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.test(data)           # run inference and evaluation

        if (i+1) % opt.display_freq == 0:
            model.display_test(i, webpage)

    webpage.save()  # save the HTML

    # print model specific evaluation results
    model.print_results(i)

    print(f"Total run time testing script: {time.time()-start_time:.1f} sec ({i} iterations)")






