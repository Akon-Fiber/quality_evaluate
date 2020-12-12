#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
from PIL import Image
from orientation_cut_analysis import OrientationCutAnalysis


class ScoreCalculation(object):
    def __init__(self, args):

        self.__orientation_cut_analysis = OrientationCutAnalysis(
            model_path=args.attribute_model_path,
            gpu_ids=args.gpu_ids
        )


    def __sobel_blur_score(self, image):
        array_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        gray_image = array_image.convert("L")
        matrix_image = np.matrix(gray_image)
        tmp = cv2.Sobel(matrix_image, cv2.CV_16U, 1, 1, ksize=3)
        score = cv2.mean(tmp)[0]
        return score


    def __get_image_size_info(self, images_via_filtering):
        height_info = list()
        width_info = list()
        for img in images_via_filtering:
            img_size = img.shape
            height_info.append(img_size[0])
            width_info.append(img_size[1])
        return height_info, width_info


    def __get_image_blur_info(self, images_via_filtering):
        blur_info = list()
        for img in images_via_filtering:
            blur_score = self.__sobel_blur_score(img)
            blur_info.append(blur_score)
        return blur_info


    def __get_image_orientation_cut_info(self, images_via_filtering):
        orientation_cut_info = None
        for index, img in enumerate(images_via_filtering):
            images_via_filtering[index] = cv2.resize(img, (224, 224))
        images_batch = [images_via_filtering[i:i + 10] for i in range(0, len(images_via_filtering), 10)]
        for batch in images_batch:
            batch = np.array(batch)
            normal_indexes = self.__orientation_cut_analysis.predict(batch)
            if orientation_cut_info is None:
                orientation_cut_info = normal_indexes
            else:
                orientation_cut_info = np.vstack((orientation_cut_info, normal_indexes))
        return orientation_cut_info


    def assessment(self, images_via_filtering):
        images_num = len(images_via_filtering)

        height_info, width_info = self.__get_image_size_info(images_via_filtering)

        blur_info = self.__get_image_blur_info(images_via_filtering)

        images_to_resize = images_via_filtering.copy()
        orientation_cut_info = self.__get_image_orientation_cut_info(images_to_resize)
        img_info = np.array([height_info, width_info, blur_info]).T
        img_info = np.hstack((orientation_cut_info, img_info))
        np.array(img_info)
        images_score = np.ones((images_num, 6)).astype("float64")

        images_score[:, 0] = abs(img_info[:, 2] / img_info[:, 3] - 2.5)

        images_score[:, 1] = img_info[:, 2] * img_info[:, 3]

        images_score[:, 4] = img_info[:, 4]
        images_score = images_score / images_score.max(axis=0)

        images_score[:, 2] = img_info[:, 0]
        images_score[:, 2][images_score[:, 2] >= 2.0] = 0.75
        images_score[:, 2][images_score[:, 2] == 1.0] = 0.5
        images_score[:, 2][images_score[:, 2] == 0.0] = 1.0

        images_score[:, 3] = img_info[:, 1]
        images_score[:, 3][images_score[:, 3] == 1.0] = 0.5
        images_score[:, 3][images_score[:, 3] == 0.0] = 1.0

        images_score[:, 5] = -0.5 * images_score[:, 0] + 1.5 * images_score[:, 1] + images_score[:, 2] + images_score[:, 3] + images_score[:, 4]
        return images_score
