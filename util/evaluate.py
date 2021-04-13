# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-03-11 11:16
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-04-13 17:26

import itertools
import torch
import numpy as np


def compute_IOU(outputs: torch.Tensor, labels: torch.Tensor, smooth=1e-6):
    # taken from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x

    outputs = outputs.bool()
    labels = labels.bool()


    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou  # Or thresholded.mean() if you are interested in average across the batch


######### COPIED FROM BASILVH GITHUB ##########

# https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
def get_powerset(iterable):
    """compute the combinations of all objects, with different subset sizes"""
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s)+1)))



def get_separate_masks(mask, num_obj):
    """Convert the colors in the mask back to binary masks per object"""
    # get width and height of image
    [_, h, w] = mask.shape

    # create the output tensor
    out = torch.empty(num_obj, h, w)

    # get specific color for every object, and extract object mask
    for i, col in enumerate(mask[2, :, :].unique()[1:]):
        object_mask = (mask[2, :, :] == col).int()
        out[i, :, :] = object_mask

    # out is a binary mask here, with one object per channel
    return out



def is_mask_success(true_masks, object_cnt, pred_mask, min_iou=0.5):
    '''
    Given a collection of ground truth masks for each individual object,
    calculates whether the predicted mask matches any possible subset.
    A successful case contributes positively to the Object Discovery Performance (ODP).
    '''

    true_masks = get_separate_masks(true_masks, object_cnt)
    true_masks = true_masks.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    true_masks = [true_masks[i] for i in range(object_cnt)] # convert to list
    tm_powerset = get_powerset(true_masks)

    # Loop over all possible subsets of ground truth objects
    for tm_subset in tm_powerset:
        # combine all indidividual object masks into one
        true_mask = (np.array(tm_subset).sum(axis=0) > 0.5)
        # compute intersection and union between GT and prediction
        intersection = np.sum(true_mask * (pred_mask > 0.5))
        union = np.sum((true_mask + pred_mask) > 0.5)

        # compute IOU, if above threshold, the mask is correct
        iou = intersection / union
        if iou >= min_iou:
            return True


    # The predicted mask does not match any subset whatsoever
    return False



######### COPIED FROM BASILVH GITHUB ##########