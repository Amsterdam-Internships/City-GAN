# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-11-13 16:47
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-11-16 14:15


import torch
import numpy as np


from base_net import BaseNet



class Generator(BaseNet):
    """
    Generator class
    """
    def __init__(self, config):
        super(Generator, self),__init__()


    def forward(self, x):
        pass