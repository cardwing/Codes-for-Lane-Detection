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
    out_logits_ref = out_logits
    out_logits = tf.nn.softmax(logits=out_logits)
    out_logits_out = tf.argmax(out_logits, axis=-1) # 8 x 288 x 800

    pred_0 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(instance_label_tensor, 0), tf.int64), tf.cast(tf.equal(out_logits_out, 0), tf.int64)))

    pred_1 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(instance_label_tensor, 1), tf.int64), tf.cast(tf.equal(out_logits_out, 1), tf.int64)))
    pred_2 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(instance_label_tensor, 2), tf.int64), tf.cast(tf.equal(out_logits_out, 2), tf.int64)))
    pred_3 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(instance_label_tensor, 3), tf.int64), tf.cast(tf.equal(out_logits_out, 3), tf.int64)))
    pred_4 = tf.count_nonzero(tf.multiply(tf.cast(tf.equal(instance_label_tensor, 4), tf.int64), tf.cast(tf.equal(out_logits_out, 4), tf.int64)))
    gt_all = tf.count_nonzero(tf.cast(tf.greater(instance_label_tensor, 0), tf.int64))
    gt_back = tf.count_nonzero(tf.cast(tf.equal(instance_label_tensor, 0), tf.int64))

    pred_all = tf.add(tf.add(tf.add(pred_1, pred_2), pred_3), pred_4)

    accuracy = tf.divide(pred_all, gt_all)
    accuracy_back = tf.divide(pred_0, gt_back)

    # Compute mIoU of Lanes
    overlap_1 = pred_1
    union_1 = tf.add(tf.count_nonzero(tf.cast(tf.equal(instance_label_tensor, 1), 
                                             tf.int64)), 
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 1), 
                                             tf.int64)))
    union_1 = tf.subtract(union_1, overlap_1)
    IoU_1 = tf.divide(overlap_1, union_1)

    overlap_2 = pred_2
    union_2 = tf.add(tf.count_nonzero(tf.cast(tf.equal(instance_label_tensor, 2), 
                                             tf.int64)), 
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 2), 
                                             tf.int64)))
    union_2 = tf.subtract(union_2, overlap_2)
    IoU_2 = tf.divide(overlap_2, union_2)

    overlap_3 = pred_3
    union_3 = tf.add(tf.count_nonzero(tf.cast(tf.equal(instance_label_tensor, 3), 
                                             tf.int64)), 
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 3), 
                                             tf.int64)))
    union_3 = tf.subtract(union_3, overlap_3)
    IoU_3 = tf.divide(overlap_3, union_3)

    overlap_4 = pred_4
    union_4 = tf.add(tf.count_nonzero(tf.cast(tf.equal(instance_label_tensor, 4), 
                                             tf.int64)), 
                     tf.count_nonzero(tf.cast(tf.equal(out_logits_out, 4), 
                                             tf.int64)))
    union_4 = tf.subtract(union_4, overlap_4)
    IoU_4 = tf.divide(overlap_4, union_4)

    IoU = tf.reduce_mean(tf.stack([IoU_1, IoU_2, IoU_3, IoU_4]))


    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.polynomial_decay(CFG.TRAIN.LEARNING_RATE, global_step, 
                                                                90100, power=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=
                                           learning_rate, momentum=0.9).minimize(loss=total_loss,
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
        train_accuracy_back_mean= []

        val_cost_time_mean = []
        val_instance_loss_mean = []
        val_existence_loss_mean = []
        val_accuracy_mean = []
        val_accuracy_back_mean = []
        val_IoU_mean = []

        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            gt_imgs, instance_gt_labels, existence_gt_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)

            gt_imgs = [cv2.resize(tmp,
                                  dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                  dst=tmp,
                                  interpolation=cv2.INTER_CUBIC)
                       for tmp in gt_imgs]
            gt_imgs = [(tmp - VGG_MEAN) for tmp in gt_imgs]

            instance_gt_labels = [cv2.resize(tmp,
                                             dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                             dst=tmp,
                                             interpolation=cv2.INTER_NEAREST)
                                  for tmp in instance_gt_labels]
            
            phase_train = 'train'

            _, c, train_accuracy, train_accuracy_back, train_instance_loss, train_existence_loss, binary_seg_img = \
                sess.run([optimizer, total_loss,
                          accuracy, accuracy_back,
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
            train_accuracy_back_mean.append(train_accuracy_back)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                print('Epoch: {:d} loss_ins= {:6f} ({:6f}) loss_ext= {:6f} ({:6f}) accuracy= {:6f} ({:6f}) accuracy_back= {:6f} ({:6f})'
                         ' mean_time= {:5f}s '.
                         format(epoch + 1, train_instance_loss, np.mean(train_instance_loss_mean), train_existence_loss, np.mean(train_existence_loss_mean), train_accuracy, np.mean(train_accuracy_mean), train_accuracy_back, np.mean(train_accuracy_back_mean), np.mean(train_cost_time_mean)))


            if epoch % 500 == 0:
                train_cost_time_mean.clear()
                train_instance_loss_mean.clear()
                train_existence_loss_mean.clear()
                train_accuracy_mean.clear()
                train_accuracy_back_mean.clear()

            if epoch % 1000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

            if epoch % 10000 != 0 or epoch == 0:
                continue

            for epoch_val in range(int(9675 / 8.0)):

                # validation part
                gt_imgs_val, instance_gt_labels_val, existence_gt_labels_val \
                  = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
                gt_imgs_val = [cv2.resize(tmp,
                                          dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                          dst=tmp,
                                          interpolation=cv2.INTER_CUBIC)
                               for tmp in gt_imgs_val]
                gt_imgs_val = [(tmp - VGG_MEAN) for tmp in gt_imgs_val]

                instance_gt_labels_val = [cv2.resize(tmp,
                                                     dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                                     dst=tmp,
                                                     interpolation=cv2.INTER_NEAREST)
                                          for tmp in instance_gt_labels_val]
                phase_val = 'test'

                t_start_val = time.time()
                c_val, val_accuracy, val_accuracy_back, val_IoU, val_instance_loss, val_existence_loss = \
                  sess.run([total_loss, accuracy, accuracy_back, IoU, instance_loss, existence_loss],
                             feed_dict={input_tensor: gt_imgs_val,
                                        instance_label_tensor: instance_gt_labels_val,
                                        existence_label_tensor: existence_gt_labels_val,
                                        phase: phase_val})

                cost_time_val = time.time() - t_start_val
                val_cost_time_mean.append(cost_time_val)
                val_instance_loss_mean.append(val_instance_loss)
                val_existence_loss_mean.append(val_existence_loss)
                val_accuracy_mean.append(val_accuracy)
                val_accuracy_back_mean.append(val_accuracy_back)
                val_IoU_mean.append(val_IoU)

                if epoch_val % 1 == 0:
                    print('Epoch_Val: {:d} loss_ins= {:6f} ({:6f}) '
                             'loss_ext= {:6f} ({:6f}) accuracy= {:6f} ({:6f}) accuracy_back= {:6f} ({:6f}) mIoU= {:6f} ({:6f})'
                             'mean_time= {:5f}s '.
                             format(epoch_val + 1, val_instance_loss, np.mean(val_instance_loss_mean), val_existence_loss, np.mean(val_existence_loss_mean), val_accuracy, np.mean(val_accuracy_mean), val_accuracy_back, np.mean(val_accuracy_back_mean), val_IoU, np.mean(val_IoU_mean), np.mean(val_cost_time_mean)))

	        
            val_cost_time_mean.clear()
            val_instance_loss_mean.clear()
            val_existence_loss_mean.clear()
            val_accuracy_mean.clear()
            val_accuracy_back_mean.clear()
            val_IoU_mean.clear()

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net)
