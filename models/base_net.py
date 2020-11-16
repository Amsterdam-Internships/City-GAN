# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-11-13 16:50
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-11-13 17:02


import torch
import torch.nn as nn



class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()

