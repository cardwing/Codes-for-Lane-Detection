#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-29 下午2:04
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : dilation_encoder.py
# @IDE: PyCharm Community Edition
"""
实现一个基于VGG16的特征编码类
"""
from collections import OrderedDict
import math

import tensorflow as tf

from encoder_decoder_model import cnn_basenet
from config import global_config

CFG = global_config.cfg


class VGG16Encoder(cnn_basenet.CNNBaseModel):
    """
    实现了一个基于vgg16的特征编码类
    """

    def __init__(self, phase):
        """

        :param phase:
        """
        super(VGG16Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _conv_dilated_stage(self, input_tensor, k_size, out_dims, name,
                            dilation=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.dilation_conv(input_tensor=input_tensor, out_dims=out_dims,
                                      k_size=k_size, rate=dilation,
                                      use_bias=False, padding=pad, name='conv')

            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _fc_stage(self, input_tensor, out_dims, name, use_bias=False):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name):
            fc = self.fullyconnect(inputdata=input_tensor, out_dim=out_dims, use_bias=use_bias,
                                   name='fc')

            bn = self.layerbn(inputdata=fc, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def encode(self, input_tensor, name):
        """
        根据vgg16框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :return: 输出vgg16编码特征
        """
        ret = OrderedDict()
        with tf.variable_scope(name):
            # conv stage 1_1
            conv_1_1 = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                        out_dims=64, name='conv1_1')

            # conv stage 1_2
            conv_1_2 = self._conv_stage(input_tensor=conv_1_1, k_size=3,
                                        out_dims=64, name='conv1_2')

            # pool stage 1
            pool1 = self.maxpooling(inputdata=conv_1_2, kernel_size=2,
                                    stride=2, name='pool1')

            # conv stage 2_1
            conv_2_1 = self._conv_stage(input_tensor=pool1, k_size=3,
                                        out_dims=128, name='conv2_1')

            # conv stage 2_2
            conv_2_2 = self._conv_stage(input_tensor=conv_2_1, k_size=3,
                                        out_dims=128, name='conv2_2')

            # pool stage 2
            pool2 = self.maxpooling(inputdata=conv_2_2, kernel_size=2,
                                    stride=2, name='pool2')

            # conv stage 3_1
            conv_3_1 = self._conv_stage(input_tensor=pool2, k_size=3,
                                        out_dims=256, name='conv3_1')

            # conv_stage 3_2
            conv_3_2 = self._conv_stage(input_tensor=conv_3_1, k_size=3,
                                        out_dims=256, name='conv3_2')

            # conv stage 3_3
            conv_3_3 = self._conv_stage(input_tensor=conv_3_2, k_size=3,
                                        out_dims=256, name='conv3_3')

            # pool stage 3
            pool3 = self.maxpooling(inputdata=conv_3_3, kernel_size=2,
                                    stride=2, name='pool3')

            # conv stage 4_1
            conv_4_1 = self._conv_stage(input_tensor=pool3, k_size=3,
                                        out_dims=512, name='conv4_1')

            # conv stage 4_2
            conv_4_2 = self._conv_stage(input_tensor=conv_4_1, k_size=3,
                                        out_dims=512, name='conv4_2')

            # conv stage 4_3
            conv_4_3 = self._conv_stage(input_tensor=conv_4_2, k_size=3,
                                        out_dims=512, name='conv4_3')

            ### add dilated convolution ###

            # conv stage 5_1
            conv_5_1 = self._conv_dilated_stage(input_tensor=conv_4_3, k_size=3,
                                                out_dims=512, dilation=2, name='conv5_1')

            # conv stage 5_2
            conv_5_2 = self._conv_dilated_stage(input_tensor=conv_5_1, k_size=3,
                                                out_dims=512, dilation=2, name='conv5_2')

            # conv stage 5_3
            conv_5_3 = self._conv_dilated_stage(input_tensor=conv_5_2, k_size=3,
                                                out_dims=512, dilation=2, name='conv5_3')

            # added part of SCNN #

            # conv stage 5_4
            conv_5_4 = self._conv_dilated_stage(input_tensor=conv_5_3, k_size=3,
                                                out_dims=1024, dilation=4, name='conv5_4')

            # conv stage 5_5
            conv_5_5 = self._conv_stage(input_tensor=conv_5_4, k_size=1,
                                        out_dims=128, name='conv5_5')  # 8 x 36 x 100 x 128

            # add message passing #

            # top to down #

            feature_list_old = []
            feature_list_new = []
            for cnt in range(conv_5_5.get_shape().as_list()[1]):
                feature_list_old.append(tf.expand_dims(conv_5_5[:, cnt, :, :], axis=1))
            feature_list_new.append(tf.expand_dims(conv_5_5[:, 0, :, :], axis=1))

            w1 = tf.get_variable('W1', [1, 9, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_1"):
                conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[1])
                feature_list_new.append(conv_6_1)

            for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_6_1", reuse=True):
                    conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[cnt])
                    feature_list_new.append(conv_6_1)

            # down to top #
            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(CFG.TRAIN.IMG_HEIGHT / 8) - 1
            feature_list_new.append(feature_list_old[length])

            w2 = tf.get_variable('W2', [1, 9, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_2"):
                conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[length - 1])
                feature_list_new.append(conv_6_2)

            for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
                with tf.variable_scope("convs_6_2", reuse=True):
                    conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[length - cnt])
                    feature_list_new.append(conv_6_2)

            feature_list_new.reverse()

            processed_feature = tf.stack(feature_list_new, axis=1)
            processed_feature = tf.squeeze(processed_feature, axis=2)

            # left to right #

            feature_list_old = []
            feature_list_new = []
            for cnt in range(processed_feature.get_shape().as_list()[2]):
                feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
            feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

            w3 = tf.get_variable('W3', [9, 1, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_3"):
                conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[1])
                feature_list_new.append(conv_6_3)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_6_3", reuse=True):
                    conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[cnt])
                    feature_list_new.append(conv_6_3)

            # right to left #

            feature_list_old = feature_list_new
            feature_list_new = []
            length = int(CFG.TRAIN.IMG_WIDTH / 8) - 1
            feature_list_new.append(feature_list_old[length])

            w4 = tf.get_variable('W4', [9, 1, 128, 128],
                                 initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
            with tf.variable_scope("convs_6_4"):
                conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                                  feature_list_old[length - 1])
                feature_list_new.append(conv_6_4)

            for cnt in range(2, processed_feature.get_shape().as_list()[2]):
                with tf.variable_scope("convs_6_4", reuse=True):
                    conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                                      feature_list_old[length - cnt])
                    feature_list_new.append(conv_6_4)

            feature_list_new.reverse()
            processed_feature = tf.stack(feature_list_new, axis=2)
            processed_feature = tf.squeeze(processed_feature, axis=3)

            #######################

            dropout_output = self.dropout(processed_feature, 0.9, is_training=self._is_training,
                                          name='dropout')  # 0.9 denotes the probability of being kept

            conv_output = self.conv2d(inputdata=dropout_output, out_channel=5,
                                      kernel_size=1, use_bias=True, name='conv_6')

            ret['prob_output'] = tf.image.resize_images(conv_output, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])

            ### add lane existence prediction branch ###

            # spatial softmax #
            features = conv_output  # N x H x W x C
            softmax = tf.nn.softmax(features)

            avg_pool = self.avgpooling(softmax, kernel_size=2, stride=2)
            _, H, W, C = avg_pool.get_shape().as_list()
            reshape_output = tf.reshape(avg_pool, [-1, H * W * C])
            fc_output = self.fullyconnect(reshape_output, 128)
            relu_output = self.relu(inputdata=fc_output, name='relu6')
            fc_output = self.fullyconnect(relu_output, 4)
            existence_output = fc_output

            ret['existence_output'] = existence_output

        return ret


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3],
                       name='input')
    encoder = VGG16Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a, name='encode')
    print(ret)
