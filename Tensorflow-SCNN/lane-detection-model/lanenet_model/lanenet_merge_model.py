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
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from lanenet_model import lanenet_discriminative_loss


class LaneNet(cnn_basenet.CNNBaseModel):
    """
    Lane detection model
    """
    def __init__(self, phase, net_flag='vgg'):
        """

        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=self._phase,
                                                       n=5)
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def _build_model(self, input_tensor, name):
        """
        feed forward
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):

            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')
            return encode_ret

    def compute_loss(self, input_tensor, binary_label, existence_label, name):
        """
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # feed forward to obtain logits

            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')

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
            weights = tf.reduce_sum(class_weights * binary_label_reshape, axis=1)
       
            binary_segmenatation_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=binary_label_reshape, logits=decode_logits_reshape, name='entropy_loss')

            # binary_segmenatation_loss = binary_segmenatation_loss * weights # weighted loss function

            binary_segmenatation_loss = tf.reduce_mean(binary_segmenatation_loss)

            # Compute the sigmoid loss

            existence_logit = inference_ret['existence_output']
            
            existence_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=existence_label, logits=existence_logit)

            existence_loss = tf.reduce_mean(existence_loss)
 
            # Compute the overall loss

            total_loss = binary_segmenatation_loss + 0.1 * existence_loss

            ret = {
                'total_loss': total_loss,
                'instance_seg_logits': decode_logits,
                'instance_seg_loss': binary_segmenatation_loss,
                'existence_logits': existence_logit,
                'existence_pre_loss': existence_loss
            }

            return ret

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):

            # feed forward to obtain logits

            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')

            # Compute loss

            decode_logits = inference_ret['prob_output']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)

            # Predict lane existence:
            existence_logit = inference_ret['existence_output']
            existence_output = tf.nn.sigmoid(existence_logit)

            return binary_seg_ret, existence_output


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    ret = model.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                             instance_label=instance_label, name='loss')
    print(ret['total_loss'])
