import tensorflow as tf
import numpy as np
##################################################################################
# for 3D CNN
##################################################################################
def normalize_tf(x, name="normalize"):
    with tf.name_scope("name"):
        x = tf.subtract(x,tf.reduce_min(x))
        return tf.divide(x,tf.reduce_max(x))

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
        print('conv3d shape : {} => {}'.format(input_.shape, output_dim))
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

##################################################################################
# from old file
##################################################################################
# def batch_norm(x, is_training=True, scope='batch_norm'):
#     # mean, var = tf.nn.moments(x, axes=0)
#     return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
#                                         center=True, scale=True, updates_collections=None,
#                                         is_training=is_training)
#     # return tf.nn.batch_normalization(x,mean=mean,variance=var,\
#     #                                  offset=0.01, scale=1, variance_epsilon=1e-05)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

#Helper functions to define weights and biases
def init_weights(shape):
    '''
    Input: shape -  this is the shape of a matrix used to represent weigts for the arbitrary layer
    Output: wights randomly generated with size = shape
    '''
    return tf.Variable(tf.truncated_normal(shape, 0, 0.05))


def init_weights_res(shape):
    '''
    Input: shape -  this is the shape of a matrix used to represent weigts for the arbitrary layer
    Output: wights randomly generated with size = shape
    '''
    return tf.Variable(tf.truncated_normal(shape, 0, 0.1))

def init_biases(shape):
    '''
    Input: shape -  this is the shape of a vector used to represent biases for the arbitrary layer
    Output: a vector for biases (all zeros) lenght = shape
    '''
    return tf.Variable(tf.zeros(shape))

def fully_connected_res_layer(inputs, input_shape, output_shape, keep_prob, activation=tf.nn.relu):
    '''
    This function is used to create tensorflow fully connected layer.

    Inputs: inputs - input data to the layer
            input_shape - shape of the inputs features (number of nodes from the previous layer)
            output_shape - shape of the layer
            activatin - used as an activation function for the layer (non-liniarity)
    Output: layer - tensorflow fully connected layer

    '''
    # definine weights and biases
    weights = init_weights_res([input_shape, output_shape])
    biases = init_biases([output_shape])
    # x*W + b <- computation for the layer values
    layer = tf.matmul(inputs, weights) + biases + inputs
    layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    # if activation argument is not None, we put layer values through an activation function
    if activation != None:
        layer = activation(layer)

    return layer

def fully_connected_layer(inputs, input_shape, output_shape, keep_prob, activation=tf.nn.relu):
    '''
    This function is used to create tensorflow fully connected layer.

    Inputs: inputs - input data to the layer
            input_shape - shape of the inputs features (number of nodes from the previous layer)
            output_shape - shape of the layer
            activatin - used as an activation function for the layer (non-liniarity)
    Output: layer - tensorflow fully connected layer

    '''
    # definine weights and biases
    weights = init_weights([input_shape, output_shape])
    biases = init_biases([output_shape])
    # x*W + b <- computation for the layer values
    layer = tf.matmul(inputs, weights) + biases
    # layer = batch_norm(layer)
    layer = tf.nn.dropout(layer, keep_prob=keep_prob)
    # if activation argument is not None, we put layer values through an activation function
    if activation != None:
        layer = activation(layer)

    return layer

import tensorflow as tf
import tensorflow.contrib as tf_contrib

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0., stddev=0.05)
# weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
weight_regularizer = None
# weight_init = tf_contrib.layers.variance_scaling_initializer()

##################################################################################
# Layer
##################################################################################
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    # print([pad,pad])
    # print(x.shape)
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            # x = tf.pad(x, [[0, 0], [pad, pad,], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)
        return x

def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        else:
            output_shape =[x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)
        return x

def fully_connected(x, units, weight_initializer, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x

def gaussian_noise_layer(train_data, std):
    noise = np.random.normal(0, std, np.shape(train_data))
    return train_data + noise


def gaussian_noise_layer_tf(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def flatten(x) :
    with tf.name_scope("flatten"):
        return tf.layers.flatten(x)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def onehot(x, depth):
    return tf.one_hot(x, depth)
##################################################################################
# Sampling
##################################################################################

def max_pooling(x,ks,s, padding="SAME"):
    '''
    3d max pooling
    '''
    with tf.name_scope("max_pool"):
        return tf.nn.max_pool3d(x, ksize=ks, strides=s, padding=padding)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)
##################################################################################
# Activation function
##################################################################################
def lrelu(x, alpha=0.2):
    with tf.name_scope("lrelu"):
        return tf.nn.leaky_relu(x, alpha)

def relu(x, scope):
    with tf.name_scope("relu"):
        return tf.nn.relu(x)

def tanh(x):
    with tf.name_scope("tanh"):
        return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm_contrib(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

##################################################################################
# Loss function
##################################################################################

def classifier_loss(loss_func, predictions, targets):
    real_loss = 0
    fake_loss = 0
    if loss_func =='L2':
        loss = tf.reduce_mean(tf.squared_difference(targets, predictions))
    elif loss_func == 'cEntropy':
        # loss = tf.losses.softmax_cross_entropy(targets, predictions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets, logits=predictions)
        loss = tf.reduce_mean(cross_entropy)

    return loss

def accuracy(predictions, labels):
    '''
    we should check whether label is already one hot vectors.
    :return:
    '''
    # print(type(tf.argmax(predictions, 1)), type(labels))
    # print(predictions.shape, labels.shape)
    # correct_pred = tf.equal(tf.argmax(predictions, 1),tf.cast(labels, dtype=tf.int64))
    correct_pred = tf.equal(tf.argmax(predictions, 1),tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, "float")) * 100
'''
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
'''

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        return x + x_init

##################################################################################
# noise addition
##################################################################################
