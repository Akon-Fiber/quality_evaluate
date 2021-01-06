#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from quality_parameters import QualityParameters


class OrientationCutAnalysis(object):
    def __init__(self, model_path, gpu_ids=[]):
        """"""
        self.__parameters = QualityParameters()
        self.__model_path = model_path
        self.__gpu_ids = gpu_ids
        self.__load_model()
        return

    def __load_model(self):
        """"""
        if len(self.__gpu_ids) != 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, self.__gpu_ids)))
            self.__parameters.device = torch.device('cuda')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.__parameters.device = torch.device('cpu')
        
        self.__parameters.load_model = self.__model_path
        if self.__parameters.arch == "MobileNetV2" and self.__model_path:
            self.__model = torch.jit.load(self.__model_path)
            self.__model = self.__model.to(self.__parameters.device)
        else:
            raise ValueError("INFO: [ {} ]: Not Exist This Architecture!".format(self.__parameters.arch))

        self.__mean = torch.Tensor(self.__parameters.mean).to(self.__parameters.device)
        self.__std = torch.Tensor(self.__parameters.std).to(self.__parameters.device)
        return

    def predict(self, imgs):

        imgs = torch.from_numpy(imgs.astype(np.float32)) 
        imgs = imgs.to(self.__parameters.device)
        imgs = imgs[:, :, :, [2, 1, 0]]
        imgs = (imgs / 255. - self.__mean) / self.__std
        imgs = imgs.permute(0, 3, 1, 2)
        scores = self.__model(imgs).data.cpu().numpy()
        normal_indexes = self.__binary2normal(scores)
        return normal_indexes
    
    def __binary2normal(self, binary_scores):
        """"""
        normal_indexes = np.zeros((len(binary_scores), len(self.__parameters.attribute_names)))
        for i in range(len(self.__parameters.attribute_indexes)-1):
            single_attribute_scores = binary_scores[:, self.__parameters.attribute_indexes[i]:self.__parameters.attribute_indexes[i+1]]
            single_attribute_indexes = np.argmax(single_attribute_scores, axis=1)
            normal_indexes[:, i] = single_attribute_indexes
        return normal_indexes
