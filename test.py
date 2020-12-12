#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from quality_evaluate import QualityEvaluate


class Parameter(object):
    def __init__(self):
        self.gpu_ids = [0]
        self.attribute_model_path = "./model/quality_evaluate_v0.0.1_478f086d5f01def75aec25ae593406bc.pth"


if __name__ == "__main__":
    filter_size = 80
    data_path = "./test_data"
    frame_shape = (1080, 1920)
    args = Parameter()
    evaluate = QualityEvaluate(args)
    imgs_cv_list = list()
    imgs_path_list = os.listdir(data_path)
    imgs_path_list.sort()
    for path in imgs_path_list:
        img = cv2.imread(os.path.join(data_path, path))
        imgs_cv_list.append(img)

    score, imgs_cv_list = evaluate.quality_evaluate(imgs_cv_list, frame_shape, filter_size)
    best_image_index = score.argmax(axis=0)[-1]
    best_image_cv2 = imgs_cv_list[best_image_index]
    best_image_score = round(score[best_image_index][-1], 2)
    cv2.imwrite("./best_score_%s.jpg" % best_image_score, best_image_cv2)
