#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_lanenet.py
# @IDE: PyCharm Community Edition
"""
训练lanenet模型
"""
import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf

try:
    from cv2 import cv2
except ImportError:
    pass

from config import global_config
from lanenet_model import lanenet_merge_model
from data_provider import lanenet_data_processor

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')

    return parser.parse_args()


def train_net(dataset_dir, weights_path=None, net_flag='vgg'):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset_file = ops.join(dataset_dir, 'train_gt.txt')
    val_dataset_file = ops.join(dataset_dir, 'val_gt.txt')

    assert ops.exists(train_dataset_file)

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    input_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                         CFG.TRAIN.IMG_WIDTH, 3],
                                  name='input_tensor')
    instance_label_tensor = tf.placeholder(dtype=tf.int64,
                                           shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                  CFG.TRAIN.IMG_WIDTH],
                                           name='instance_input_label')
    existence_label_tensor = tf.placeholder(dtype=tf.float32,
                                           shape=[CFG.TRAIN.BATCH_SIZE, 4],
                                           name='existence_input_label')
    phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

    net = lanenet_merge_model.LaneNet(net_flag=net_flag, phase=phase)

    # calculate the loss
    compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=instance_label_tensor,
                                   existence_label=existence_label_tensor, name='lanenet_loss')
    total_loss = compute_ret['total_loss']
    instance_loss = compute_ret['instance_seg_loss']
    existence_loss = compute_ret['existence_pre_loss']
    existence_logits = compute_ret['existence_logits']

    # calculate the accuracy
    out_logits = compute_ret['instance_seg_logits']
    out_logits = tf.nn.softmax(logits=out_logits)
    out_logits_out = tf.argmax(out_logits, axis=-1)
    out = tf.argmax(out_logits, axis=-1)
    out = tf.expand_dims(out, axis=-1)

    idx = tf.where(tf.equal(instance_label_tensor, 1))
    pix_cls_ret = tf.gather_nd(out, idx)
    accuracy = tf.count_nonzero(pix_cls_ret)
    accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                               5000, 0.96, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=
                                           learning_rate).minimize(loss=total_loss,
                                                                   var_list=tf.trainable_variables(),
                                                                   global_step=global_step)

    # Set tf saver
    saver = tf.train.Saver()
    model_save_dir = 'model/culane_lanenet/culane_scnn'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'culane_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 4}) # device_count={'GPU': 1}
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)


    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # 加载预训练参数
        if net_flag == 'vgg' and weights_path is None:
            pretrained_weights = np.load(
                './data/vgg16.npy',
                encoding='latin1').item()

            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

        train_cost_time_mean = []
        train_instance_loss_mean = []
        train_existence_loss_mean = []
        train_accuracy_mean= []

        val_cost_time_mean = []
        val_instance_loss_mean = []
        val_existence_loss_mean = []
        val_accuracy_mean = []


        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            gt_imgs, instance_gt_labels, existence_gt_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)
            gt_imgs = [cv2.resize(tmp,
                                  dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                  dst=tmp,
                                  interpolation=cv2.INTER_LINEAR)
                       for tmp in gt_imgs]
            gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]

            instance_gt_labels = [cv2.resize(tmp,
                                             dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                             dst=tmp,
                                             interpolation=cv2.INTER_NEAREST)
                                  for tmp in instance_gt_labels]
            


            phase_train = 'train'

            _, c, train_accuracy, train_instance_loss, train_existence_loss, binary_seg_img = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          instance_loss,
                          existence_loss,
                          out_logits_out],
                         feed_dict={input_tensor: gt_imgs,
                                    instance_label_tensor: instance_gt_labels,
                                    existence_label_tensor: existence_gt_labels,
                                    phase: phase_train})

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            train_instance_loss_mean.append(train_instance_loss)
            train_existence_loss_mean.append(train_existence_loss)
            train_accuracy_mean.append(train_accuracy)

            # validation part
            gt_imgs_val, instance_gt_labels_val, existence_gt_labels_val \
                = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
            gt_imgs_val = [cv2.resize(tmp,
                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                      dst=tmp,
                                      interpolation=cv2.INTER_LINEAR)
                           for tmp in gt_imgs_val]
            gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
            instance_gt_labels_val = [cv2.resize(tmp,
                                                 dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                 dst=tmp,
                                                 interpolation=cv2.INTER_NEAREST)
                                      for tmp in instance_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            c_val, val_accuracy, val_instance_loss, val_existence_loss = \
                sess.run([total_loss, accuracy, instance_loss, existence_loss],
                         feed_dict={input_tensor: gt_imgs_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    existence_label_tensor: existence_gt_labels_val,
                                    phase: phase_val})

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)
            val_instance_loss_mean.append(val_instance_loss)
            val_existence_loss_mean.append(val_existence_loss)
            val_accuracy_mean.append(val_accuracy)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                print('Epoch: {:d} loss_ins= {:6f} ({:6f}) loss_ext= {:6f} ({:6f}) accuracy= {:6f} ({:6f})'
                         ' mean_time= {:5f}s '.
                         format(epoch + 1, train_instance_loss, np.mean(train_instance_loss_mean), train_existence_loss, np.mean(train_existence_loss_mean), train_accuracy, np.mean(train_accuracy_mean), np.mean(train_cost_time_mean))) # log.info

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                print('Epoch_Val: {:d} loss_ins= {:6f} ({:6f}) '
                         'loss_ext= {:6f} ({:6f}) accuracy= {:6f} ({:6f})'
                         'mean_time= {:5f}s '.
                         format(epoch + 1, val_instance_loss, np.mean(val_instance_loss_mean), val_existence_loss, np.mean(val_existence_loss_mean), val_accuracy, np.mean(val_accuracy_mean),
                                np.mean(val_cost_time_mean)))

            if epoch % 500 == 0:
                train_cost_time_mean.clear()
                train_instance_loss_mean.clear()
                train_existence_loss_mean.clear()
                train_accuracy_mean.clear()

                val_cost_time_mean.clear()
                val_instance_loss_mean.clear()
                val_existence_loss_mean.clear()
                val_accuracy_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)
