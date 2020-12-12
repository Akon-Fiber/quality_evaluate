#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from score_calculation import ScoreCalculation
logger = logging.getLogger()


class QualityEvaluate(object):
    def __init__(self, args):
        # 综合评分模块初始化
        self.__score_calculation = ScoreCalculation(args)

    # 质量评估
    def quality_evaluate(self, images_cv_input, frame_size, filter_size):
        """
        :param images_cv_input: 行人小图输入，[img1, img2, …] 其中img1，img2，为cv格式
        :param frame_size: 原图尺寸
        :param filter_size: 过滤尺寸
        :return: images_score: 行人小图评分 格式 array[score1, score2, …]
        :return images_via_filtering: 过滤后剩余的行人图片, [img1, img2, …] 其中img1，img2，为cv格式
        """
        # 图片过滤
        images_via_filtering = self.__abnormal_images_filter(images_cv_input, frame_size, filter_size)
        images_num = len(images_via_filtering)
        if images_num == 0:
            return None, images_via_filtering
        # 综合评分
        images_score = self.__score_calculation.assessment(images_via_filtering)
        return images_score, images_via_filtering

    # 过滤模块：过滤异常图片、异常尺寸、异常高宽比
    @staticmethod
    def __abnormal_images_filter(images_cv_input, frame_size, filter_size):
        print("--- Total Number of Images: {}".format(len(images_cv_input)))
        filtered_img_index = list()
        images_via_filtering = list()
        for index, img in enumerate(images_cv_input):
            # 获取图片尺寸
            img_size = img.shape
            img_h = img_size[0]
            img_w = img_size[1]
            img_proportion = 0.6 * frame_size[1]
            # 过滤条件
            filter_condition = [img.shape[1] == 0, img.shape[0] == 0,  # 异常图片
                                max(img_h, img_w) >= img_proportion, min(img_h, img_w) <= filter_size,  # 异常尺寸
                                img_h / img_w >= 4, img_h / img_w <= 1]  # 高宽比异常
            # 如果满足filter_condition中的一条，过滤该图片
            for condition in filter_condition:
                if condition:
                    filtered_img_index.append(index)
                    break
            if index not in filtered_img_index:
                images_via_filtering.append(images_cv_input[index])
        return images_via_filtering
