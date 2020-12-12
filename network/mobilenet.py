#! /usr/bin/env python
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .base_model import mobilenet_v2


class MobileNetV2(nn.Module):
    def __init__(self, **kwargs):
        super(MobileNetV2, self).__init__()
        if 'num_att' in kwargs:
            self.num_att = kwargs['num_att']
        else:
            self.num_att = 35
        if 'last_conv_stride' in kwargs:
            self.last_conv_stride = kwargs['last_conv_stride']
        else:
            self.last_conv_stride = 2
        self.base = mobilenet_v2()
        self.classifier = nn.Linear(1280, 6)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
