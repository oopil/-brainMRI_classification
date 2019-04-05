import tensorflow as tf
import numpy as np
def batch_norm(x, is_training=True, scope='batch_norm'):
    # mean, var = tf.nn.moments(x, axes=0)
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training)
    # return tf.nn.batch_normalization(x,mean=mean,variance=var,\
    #                                  offset=0.01, scale=1, variance_epsilon=1e-05)

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

