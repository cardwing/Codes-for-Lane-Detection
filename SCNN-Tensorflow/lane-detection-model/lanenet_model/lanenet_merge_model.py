#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:28
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_merge_model.py
# @IDE: PyCharm Community Edition
"""
Build Lane detection model
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import cnn_basenet
from config import global_config

CFG = global_config.cfg


class LaneNet(cnn_basenet.CNNBaseModel):
    """
    Lane detection model
    """

    @staticmethod
    def inference(input_tensor, phase):
        """
        feed forward
        :param input_tensor:
        :param phase:
        :return:
        """
        encoder = vgg_encoder.VGG16Encoder(phase=phase)
        encode_ret = encoder.encode(input_tensor=input_tensor)
        return encode_ret

    @staticmethod
    def loss(inference, binary_label, existence_label):
        """
        :param inference:
        :param existence_label:
        :param binary_label:
        :return:
        """
        # feed forward to obtain logits

        inference_ret = inference

        # Compute the segmentation loss

        decode_logits = inference_ret['prob_output']
        decode_logits_reshape = tf.reshape(
            decode_logits,
            shape=[decode_logits.get_shape().as_list()[0],
                   decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
                   decode_logits.get_shape().as_list()[3]])

        binary_label_reshape = tf.reshape(
            binary_label,
            shape=[binary_label.get_shape().as_list()[0],
                   binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
        binary_label_reshape = tf.one_hot(binary_label_reshape, depth=5)
        class_weights = tf.constant([[0.4, 1.0, 1.0, 1.0, 1.0]])
        weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
        binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                                   logits=decode_logits_reshape,
                                                                   weights=weights_loss)
        binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

        # Compute the sigmoid loss

        existence_logits = inference_ret['existence_output']
        existence_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=existence_label, logits=existence_logits)
        existence_loss = tf.reduce_mean(existence_loss)

        # Compute the overall loss

        total_loss = binary_segmentation_loss + 0.1 * existence_loss
        ret = {
            'total_loss': total_loss,
            'instance_seg_logits': decode_logits,
            'instance_seg_loss': binary_segmentation_loss,
            'existence_logits': existence_logits,
            'existence_pre_loss': existence_loss
        }

        return ret
