#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午4:58
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的数据解析类
"""
import tensorflow as tf

from config import global_config

CFG = global_config.cfg
VGG_MEAN = [123.68, 116.779, 103.939]


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file, batch_size):
        """

        :param dataset_info_file:
        """
        self._dataset_info_file = dataset_info_file
        self._batch_size = batch_size
        self._img_list = self._init_dataset()
        self._next_batch_loop_count = 0

    def __len__(self):
        return self._len

    def _init_dataset(self):
        """
        :return:
        """
        img_list = []

        if not tf.gfile.Exists(self._dataset_info_file):
            raise ValueError('Failed to find file: ' + self._dataset_info_file)

        with open(self._dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                img_list.append(info_tmp[0][1:])

        self._len = len(img_list)

        return img_list

    @staticmethod
    def process_img(img_path):
        img_raw = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH],
                                             method=tf.image.ResizeMethod.BICUBIC)
        img_casted = tf.cast(img_resized, tf.float32)
        return tf.subtract(img_casted, VGG_MEAN)

    def next_batch(self):
        """
        :return:
        """

        idx_start = self._batch_size * self._next_batch_loop_count
        idx_end = self._batch_size * self._next_batch_loop_count + self._batch_size

        if idx_end > len(self):
            idx_end = len(self)

        img_list = self._img_list[idx_start:idx_end]
        self._next_batch_loop_count += 1
        return img_list
