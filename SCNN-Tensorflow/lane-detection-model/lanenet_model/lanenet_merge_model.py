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
    def inference(input_tensor, phase, name):
        """
        feed forward
        :param name:
        :param input_tensor:
        :param phase:
        :return:
        """
        with tf.variable_scope(name):
            with tf.variable_scope('inference'):
                encoder = vgg_encoder.VGG16Encoder(phase=phase)
                encode_ret = encoder.encode(input_tensor=input_tensor, name='encode')

            return encode_ret

    @staticmethod
    def test_inference(input_tensor, phase, name):
        inference_ret = LaneNet.inference(input_tensor, phase, name)
        with tf.variable_scope(name):
            # feed forward to obtain logits
            # Compute loss

            decode_logits = inference_ret['prob_output']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            prob_list = []
            kernel = tf.get_variable('kernel', [9, 9, 1, 1], initializer=tf.constant_initializer(1.0 / 81),
                                     trainable=False)

            with tf.variable_scope("convs_smooth"):
                prob_smooth = tf.nn.conv2d(tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, 0], axis=3), tf.float32),
                                           kernel, [1, 1, 1, 1], 'SAME')
                prob_list.append(prob_smooth)

            for cnt in range(1, binary_seg_ret.get_shape().as_list()[3]):
                with tf.variable_scope("convs_smooth", reuse=True):
                    prob_smooth = tf.nn.conv2d(
                        tf.cast(tf.expand_dims(binary_seg_ret[:, :, :, cnt], axis=3), tf.float32), kernel, [1, 1, 1, 1],
                        'SAME')
                    prob_list.append(prob_smooth)
            processed_prob = tf.stack(prob_list, axis=4)
            processed_prob = tf.squeeze(processed_prob, axis=3)
            binary_seg_ret = processed_prob

            # Predict lane existence:
            existence_logit = inference_ret['existence_output']
            existence_output = tf.nn.sigmoid(existence_logit)

            return binary_seg_ret, existence_output

    @staticmethod
    def loss(inference, binary_label, existence_label, name):
        """
        :param name:
        :param inference:
        :param existence_label:
        :param binary_label:
        :return:
        """
        # feed forward to obtain logits

        with tf.variable_scope(name):

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

        tf.add_to_collection('total_loss', total_loss)
        tf.add_to_collection('instance_seg_logits', decode_logits)
        tf.add_to_collection('instance_seg_loss', binary_segmentation_loss)
        tf.add_to_collection('existence_logits', existence_logits)
        tf.add_to_collection('existence_pre_loss', existence_loss)

        return ret
