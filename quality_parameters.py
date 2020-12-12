#! /usr/bin/env python
# -*- coding: utf-8 -*-


class QualityParameters(object):

    def __init__(self):
        
        self.resize = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
       
        self.attribute_dict = {
            
            "Body orientation":
                ['front', 'back', 'left', 'right'],
            "cut":
                ['no cut', 'cut']
        }
        self.attribute_names = ["Body orientation", "cut"]
        self.attribute_dict_cn = ["朝向", "截断"]
        self.attribute_names_cn = {
            "朝向":
                ['正向', '反向', '左侧', '右侧'],
            "截断":
                ['无截断', '有截断']
        }
        self.attribute_indexes = [0, 4, 6]
        self.arch = "MobileNetV2"
        self.model_kwargs = dict()
        self.model_kwargs['num_att'] = 6
        self.model_kwargs['last_conv_stride'] = 2
        return
