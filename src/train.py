# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-11-16 13:45
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-11-16 13:59


import torch
import torch.nn as nn
import numpy
import argparse
import os

import sys

# import models
from generator import Generator
from discriminator import Discriminator








def train(config):
    pass



def eval_on_test(config):
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--input_dim', type=int, default=4096,
                        help='dimensionality of the input images')
    parser.add_argument('--D_lr', type=float, default=0.0004,
                        help='learning rate Discriminator')
    parser.add_argument('--G_lr', type=float, default=0.0001,
                        help='learning rate Generator')
    parser.add_argument('--D_steps_per_G', type=int, default=4,
                        help='number of update steps for D per update of G')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument("--save_image_interval", type=int, default=100,
                        help="interval to save generated images")
    config = parser.parse_args()

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.start_epoch = 0

    print(f"Device: {config.device}")

    os.makedirs("images", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    main(config)