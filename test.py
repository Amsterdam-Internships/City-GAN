# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-03-09 15:00
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-05-18 15:51

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
    if opt.batch_size != 1:
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

    # init list to keep track of IOUs
    # IOU_list = []
    # IOU_list_eroded = []
    total_success_masks, total_n_obj, total_n_obj_recognized = 0, 0, 0
    fractions_recognized = []

    model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.test(data)           # run inference

        # compute IOU
        # IOU_batch = evaluate.compute_IOU(model.g_mask_binary, model.bin_gt)
        if opt.model == "copy"
        mask_success, used_mask_gt, n_obj = evaluate.is_mask_success(model.gt_og[0], model.gt_num_obj[0], model.g_mask_binary[0], min_iou=opt.min_iou)
        total_success_masks += mask_success
        total_n_obj += model.gt_num_obj[0]
        total_n_obj_recognized += n_obj
        fractions_recognized.append(n_obj/model.gt_num_obj[0])

        # IOU_list.extend(IOU_batch)

        # IOU_eroded = evaluate.compute_IOU(model.eroded_mask, model.bin_gt)
        # IOU_list_eroded.extend(IOU_eroded)


        if (i+1) % opt.display_freq == 0 and opt.model=="copy":
            # iou= f"{IOU_batch[0]:.2f} / {IOU_eroded[0]:.2f}, num_obj={model.gt_num_obj[0].item()}, succes: {mask_success}"

            # set used ground truth mask as visual
            model.used_comb_gt = used_mask_gt
            # add visuals to webpage
            visuals = model.get_current_visuals()  # get image results
            msg = f"num objects: {model.gt_num_obj[0].item()}, success: {mask_success} ({n_obj})"
            save_images(webpage, visuals, image_path=str(i), aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, score=msg)
    webpage.save()  # save the HTML

    # print IOU information
    # mean, min_, max_, std = np.mean(IOU_list), np.min(IOU_list), np.max(IOU_list), np.std(IOU_list)

    # print(f"\nMean: {mean:.2f}\nMin: {min_:.2f}\nMax: {max_:.2f}\nSTD: {std:.2f}")

    # mean, min_, max_, std = np.mean(IOU_list_eroded), np.min(IOU_list_eroded), np.max(IOU_list_eroded), np.std(IOU_list_eroded)
#
    # print(f"\n\nEroded: \nMean: {mean:.2f}\nMin: {min_:.2f}\nMax: {max_:.2f}\nSTD: {std:.2f}")
    if opt.model == "copy":
        ODP = total_success_masks / i
        # recognized_fraction = total_n_obj_recognized/total_n_obj
        recognized_fraction = np.mean(fractions_recognized)

        print(f"Arandjelovic score: total number of masks: {i}, succesfull: {total_success_masks}, ODP: {(ODP * 100):.1f}%")
        print(f"{total_n_obj_recognized}/{total_n_obj} objects are recognized ({recognized_fraction*100:.1f}%)")
        print(f"Total run time: {time.time()-start_time:.1f} sec")
    elif opt.model == "classifier":
        print(f"overall accuracy: {model.get_accuracies():.2f}")
        print(f"Confusion matrix: {model.confusion_matrix}")






