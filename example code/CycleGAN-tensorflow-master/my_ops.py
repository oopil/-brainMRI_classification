import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[4]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv3d(input_, output_dim, ks=4, s=(2,2,2), stddev=0.02, padding='SAME', name="conv3d"):
    with tf.variable_scope(name):
        # print('conv3d input shape : ', input_.shape)
        # print('conv3d output dimension : ', output_dim)
        return tf.layers.conv3d(input_, output_dim, ks, s, padding=padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            bias_initializer=None)

def deconv3d(input_, output_dim, ks=4, s=(2,2,2), stddev=0.02, name="deconv3d"):
    with tf.variable_scope(name):
        # print('deconv3d input shape : ', input_.shape)
        # print('deconv3d output dimension : ', output_dim)
        return tf.layers.conv3d_transpose(input_, output_dim, ks, s, padding='SAME',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    bias_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
