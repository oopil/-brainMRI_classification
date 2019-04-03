#!/usr/bin/env python
# coding: utf-8

# In[31]:


import sys
import tensorflow as tf
import pandas as pd
import argparse
sys.path.append('/home/sp/PycharmProjects/chosun_AD')
from data import *


# In[32]:


def parse_args()->argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_option_index', type=int, default='0')
    parser.add_argument('--ford_num', type=int, default='5')
    parser.add_argument('--ford_index', type=int, default='0')
    parser.add_argument('--keep_prob', type=float, default='0.9')
    parser.add_argument('--lr', type=float, default='0.01')
    parser.add_argument('--epochs', type=int, default='2000')
    parser.add_argument('--save_freq', type=int, default='300')
    parser.add_argument('--print_freq', type=int, default='100')
    
    return parser.parse_args()


# In[33]:

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


# In[34]:


# splitting the dataset to the training set and the testing set
# hyper parameters
'''
when using the all 3 options to the features, 
I could observe high training speed and high testing accuracy.
'''
# args = parse_args()

is_merge = True # True
option_num = 0 # P V T options
'''
I should set the class options like
NC vs AD
NC vs MCI
MCI vs AD

NC vs MCI vs AD
'''
class_option = ['NC vs AD','NC vs MCI','MCI vs AD','NC vs MCI vs AD']
class_option_index = 0
class_num = class_option_index//3 + 2
# SMOTEENN and SMOTETomek is not good in FCN
sampling_option = 'SMOTE' # None ADASYN SMOTE SMOTEENN SMOTETomek

ford_num = 5
ford_index = 0
keep_prob = 0.9 # 0.9

learning_rate = 0.05
epochs = 2000
print_freq = 200
save_freq = 200

'''
ford_num = args.ford_num
ford_index = args.ford_index
keep_prob = args.keep_prob # 0.9

learning_rate = args.lr
epochs = args.epochs
print_freq = args.print_freq
save_freq = args.save_freq
'''
'''
Log

1. batch normalization seems to have no positive effect on validation.
'''
# batch_size = 50


# In[35]:


data, label = dataloader(class_option[class_option_index], option_num, is_merge=is_merge)

# assert False
data, label = shuffle_two_arrays(data, label)
X_train, Y_train, X_test, Y_test = split_train_test(data, label, ford_num, ford_index)
# print(len(data[0]), len(X_train[0]))
X_train, Y_train = over_sampling(X_train, Y_train, sampling_option)
X_test, Y_test = valence_class(X_test, Y_test, class_num)
train_num = len(Y_train)
test_num = len(Y_test)
feature_num = len(X_train[0])
print(X_train.shape, X_test.shape)
# assert False


# In[36]:


graph = tf.Graph()
with graph.as_default():
    #Tensorflow placeholders - inputs to the TF graph
    inputs =  tf.placeholder(tf.float32, [None, feature_num], name='Inputs')
    targets =  tf.placeholder(tf.float32, [None, class_num], name='Targets')

    layer = [512, 1024, 2048, 1024]
    layer_last = 256
    with tf.name_scope("FCN"):
        #defining the network
        with tf.name_scope("layer1"):
            l1 = fully_connected_layer(inputs, feature_num, layer[0], keep_prob)
            l2 = fully_connected_layer(l1, layer[0], layer[0], keep_prob)
            l3 = fully_connected_layer(l2, layer[0], layer[1], keep_prob)
        with tf.name_scope("layer2"):
            l4 = fully_connected_layer(l3, layer[1], layer[1], keep_prob)
            l5 = fully_connected_layer(l4, layer[1], layer[1], keep_prob)
            l6 = fully_connected_layer(l5, layer[1], layer[1], keep_prob)
            l7 = fully_connected_layer(l6, layer[1], layer[1], keep_prob)
            l8 = fully_connected_layer(l7, layer[1], layer[1], keep_prob)
        with tf.name_scope("layer5"):
            l19 = fully_connected_layer(l8, layer[1], layer[1], keep_prob)
            l20 = fully_connected_layer(l19, layer[1], layer[1], keep_prob)
            l21 = fully_connected_layer(l20, layer[1], layer[1], keep_prob)
            l22 = fully_connected_layer(l21, layer[1], layer[1], keep_prob)
            l23 = fully_connected_layer(l22, layer[1], layer[1], keep_prob)
            l24 = fully_connected_layer(l23, layer[1], layer_last, keep_prob)
            l_fin = fully_connected_layer(l24, layer_last, class_num, activation=None, keep_prob=keep_prob)

        #defining special parameter for our predictions - later used for testing
        predictions = tf.nn.sigmoid(l_fin)

    #Mean_squared_error function and optimizer choice - Classical Gradient Descent
    cost = loss2 = tf.reduce_mean(tf.squared_difference(targets, predictions))
    tf.summary.scalar("cost", cost)
    merged_summary = tf.summary.merge_all()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # Starting session for the graph
    top_train_accur = 0
    top_test_accur = 0
    train_accur_list = []
    test_accur_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('./my_graph', graph=tf.get_default_graph())
        # writer = tf.train.SummaryWriter('./my_graph', graph=tf.get_default_graph())
        writer.add_graph(sess.graph)
        # TRAINING PORTION OF THE SESSION
        # one hot encoding
        Y_train = pd.get_dummies(Y_train)
        Y_train = np.array(Y_train)
        Y_test = pd.get_dummies(Y_test)
        Y_test = np.array(Y_test)
        for i in range(epochs):
            '''
            idx = np.random.choice(len(X_train), batch_size, replace=True)
            x_batch = X_train[idx, :]
            y_batch = Y_train[idx]
            y_batch = np.reshape(y_batch, (len(y_batch), 1))
            '''
            y_batch = Y_train
            x_batch = X_train

            summary, batch_loss, opt, preds_train = sess.run([merged_summary, cost, optimizer, predictions],                                                              feed_dict={inputs: x_batch, targets: y_batch})
            writer.add_summary(summary, global_step=i)
            train_accur = accuracy(preds_train, Y_train)
            # TESTING PORTION OF THE SESSION
            preds = sess.run([predictions], feed_dict={inputs: X_test})
            # preds_nparray = np.squeeze(np.array(preds), 0)
            preds_nparray = np.squeeze(np.array(preds), 0)
            test_accur = accuracy(preds_nparray, Y_test)

            if i % save_freq == 0:
                train_accur_list.append(train_accur//1)
                test_accur_list.append(test_accur//1)

            if i % print_freq == 0:
                # if i > (epochs//2):
                if i >= (epochs//2) and top_train_accur < train_accur:
                    top_train_accur = train_accur
                if i >= (epochs//2) and top_test_accur < test_accur:
                    top_test_accur = test_accur
                    print(top_test_accur)
                print('='*50)
                print('epoch                : ',i, '/',epochs)
                print('batch loss           : ',batch_loss)
                print("Training Accuracy (%): ", train_accur)
                print("Test Accuracy     (%): ", test_accur)
                print('pred                 :', np.transpose(np.argmax(preds_nparray, 1)))
                print('label                :', np.transpose(np.argmax(Y_test, 1)))

        writer.close()
        print('<< top accuracy >>')
        print('Training : ', top_train_accur)
        print('Testing  : ', top_test_accur)

        for i in range(len(train_accur_list)):
            print(train_accur_list[i] , test_accur_list[i])


# In[37]:

assert False
import os
line_length = 100
# is_remove_result_file = True
is_remove_result_file = False
result_file_name = '/home/sp/PycharmProjects/chosun_AD/chosun_MRI_excel_AD_classification_result'
if is_remove_result_file:
    os.system('rm {}'.format(result_file_name))
contents = []
contents.append('='*line_length + '\n')
contents.append('class option : {:30} ford index / num : {}/{:<10} train and test number : {:10} / {:<10} oversample : {}\n'    .format(class_option[class_option_index], ford_index, ford_num, train_num, test_num, sampling_option)    + 'keep probability : {:<30} epoch : {:<30} learning rate : {:<30}\n'    .format(keep_prob, epochs, learning_rate))
contents.append('top Train : {:<10} {}\n'    .format(top_train_accur//1, train_accur_list)
    + 'top Test  : {:<10} {}\n'\
    .format(top_test_accur//1, test_accur_list))

file = open(result_file_name, 'a+t')
file.writelines(contents)
# print(contents)
file.close()

'''
top_train_accur = 0
top_test_accur = 0
train_accur_list = []
test_accur_list = []
'''


# In[38]:


file = open(result_file_name, 'rt')
lines = file.readlines()
for line in lines:
    print(line)
file.close()


# In[39]:


print(1)


# In[40]:


'''
keep_prob = 0.9 # 0.9
learning_rate = 0.01
epochs = 2000
print_freq = 200
save_freq = 200

layer = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
    layer_last = 256
    with tf.name_scope("FCN"):
        #defining the network
        with tf.name_scope("layer1"):
            l1 = fully_connected_layer(inputs, feature_num, layer[0], keep_prob)
            l2 = fully_connected_layer(l1, layer[0], layer[1], keep_prob)
            l3 = fully_connected_layer(l2, layer[1], layer[2], keep_prob) + l2

        with tf.name_scope("layer2"):
            l4 = fully_connected_layer(l3, layer[2], layer[3], keep_prob)
            l5 = fully_connected_layer(l4, layer[3], layer[4], keep_prob)
            l6 = fully_connected_layer(l5, layer[4], layer[5], keep_prob)
            l7 = fully_connected_layer(l6, layer[5], layer[6], keep_prob)
            l8 = fully_connected_layer(l7, layer[6], layer[7], keep_prob) + l4
        with tf.name_scope("layer3"):
            l9 = fully_connected_layer(l8, layer[2], layer[3], keep_prob)
            l10 = fully_connected_layer(l9, layer[3], layer[4], keep_prob)
            l11 = fully_connected_layer(l10, layer[4], layer[5], keep_prob)
            l12 = fully_connected_layer(l11, layer[5], layer[6], keep_prob)
            l13 = fully_connected_layer(l12, layer[6], layer[7], keep_prob) + l9
        with tf.name_scope("layer4"):
            l14 = fully_connected_layer(l13, layer[2], layer[3], keep_prob)
            l15 = fully_connected_layer(l14, layer[3], layer[4], keep_prob)
            l16 = fully_connected_layer(l15, layer[4], layer[5], keep_prob)
            l17 = fully_connected_layer(l16, layer[5], layer[6], keep_prob)
            l18 = fully_connected_layer(l17, layer[6], layer[7], keep_prob) + l14
        with tf.name_scope("layer5"):
            l19 = fully_connected_layer(l18, layer[2], layer[2], keep_prob)
            l20 = fully_connected_layer(l19, layer[3], layer[3], keep_prob)
            l21 = fully_connected_layer(l20, layer[4], layer[4], keep_prob)
            l22 = fully_connected_layer(l21, layer[5], layer[5], keep_prob)
            l23 = fully_connected_layer(l22, layer[6], layer[6], keep_prob) + l19
            l24 = fully_connected_layer(l23, layer[6], layer_last, keep_prob)
            l_fin = fully_connected_layer(l24, layer_last, class_num, activation=None, keep_prob=keep_prob)

        #defining special parameter for our predictions - later used for testing
        predictions = tf.nn.sigmoid(l_fin)

'''
'''
keep_prob = 0.9 # 0.9

learning_rate = 0.01
epochs = 2000
print_freq = 200
save_freq = 200

    layer = [512, 1024, 2048, 1024]
    layer_last = 256
    with tf.name_scope("FCN"):
        #defining the network
        with tf.name_scope("layer1"):
            l1 = fully_connected_layer(inputs, feature_num, layer[0], keep_prob)
            l2 = fully_connected_layer(l1, layer[0], layer[0], keep_prob)# + l1
            l3 = fully_connected_layer(l2, layer[0], layer[0], keep_prob) + l2

        with tf.name_scope("layer2"):
            l4 = fully_connected_layer(l3, layer[0], layer[0], keep_prob)# + l3
            l5 = fully_connected_layer(l4, layer[0], layer[0], keep_prob)# + l4# + l3
            l6 = fully_connected_layer(l5, layer[0], layer[0], keep_prob)# + l5# + l4 + l3
            l7 = fully_connected_layer(l6, layer[0], layer[0], keep_prob)# + l6# + l5 + l4 + l3
            l8 = fully_connected_layer(l7, layer[0], layer[0], keep_prob) + l5# + l7# + l6 + l5 + l4 + l3
        with tf.name_scope("layer3"):
            l9 = fully_connected_layer(l8, layer[0], layer[1], keep_prob)# + l8
            l10 = fully_connected_layer(l9, layer[1], layer[1], keep_prob)# + l9# + l8
            l11 = fully_connected_layer(l10, layer[1], layer[1], keep_prob)# + l10# + l9 + l8
            l12 = fully_connected_layer(l11, layer[1], layer[1], keep_prob)# + l11# + l10 + l9 + l8
            l13 = fully_connected_layer(l12, layer[1], layer[1], keep_prob) + l10# + l12# + l11 + l10 + l9 + l8
        with tf.name_scope("layer4"):
            l14 = fully_connected_layer(l13, layer[1], layer[2], keep_prob)# + l13
            l15 = fully_connected_layer(l14, layer[2], layer[2], keep_prob)# + l14# + l13
            l16 = fully_connected_layer(l15, layer[2], layer[2], keep_prob)# + l15# +l14 + l13
            l17 = fully_connected_layer(l16, layer[2], layer[2], keep_prob)# + l16# + l15 + l14 + l13
            l18 = fully_connected_layer(l17, layer[2], layer[2], keep_prob) + l15# + l17# + l16 + l15 + l14 + l13
        with tf.name_scope("layer5"):
            l19 = fully_connected_layer(l18, layer[2], layer[3], keep_prob)# + l18
            l20 = fully_connected_layer(l19, layer[3], layer[3], keep_prob)# + l19# + l18
            l21 = fully_connected_layer(l20, layer[3], layer[3], keep_prob)# + l20# + l19 + l18
            l22 = fully_connected_layer(l21, layer[3], layer[3], keep_prob)# + l21# + l20 + l19 + l18
            l23 = fully_connected_layer(l22, layer[3], layer[3], keep_prob) + l20# + l22# + l21 + l20 + l19 + l18
            l24 = fully_connected_layer(l23, layer[3], layer_last, keep_prob)# + l23
            l_fin = fully_connected_layer(l24, layer_last, class_num, activation=None, keep_prob=keep_prob)

        #defining special parameter for our predictions - later used for testing
        predictions = tf.nn.sigmoid(l_fin)

'''
