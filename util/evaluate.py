# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-03-11 11:16
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-03-11 12:00


import torch


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