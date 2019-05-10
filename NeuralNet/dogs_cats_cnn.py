import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import os
import pandas as pd
from NeuralNet.CNN_data import *
from NeuralNet.NN_ops import *
from sklearn.utils import shuffle
from data_merge import *

#%%

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%%


def read_cnn_data():
    base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
    excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    diag_type = 'clinic'
    class_option = 'CN vs AD'
    class_num = 2
    test_num = 20
    fold_num = 5
    is_split_by_num = False
    sampling_option = "RANDOM"
    whole_set = CNN_dataloader(base_folder_path, diag_type, class_option, excel_path, test_num, fold_num, is_split_by_num)
    train_data, train_label, test_data, test_label = whole_set[0]
    test_data, test_label = valence_class(test_data, test_label, class_num)
    if sampling_option != "None":
        train_data, train_label = over_sampling(train_data, train_label, sampling_option)
    return train_data, one_hot_pd(train_label), test_data, one_hot_pd(test_label)
    # print(train_data)
    # print(type(train_data))
#%%
print()
print("Loading data...")
print()
train_data, train_label, val_data, val_label = read_cnn_data()
# train_label = pd.get_dummies(train_label)
print()
print("train data: {}".format(train_data.shape))
print("train label: {}".format(train_label.shape))
print()
print("validation data: {}".format(val_data.shape))
print("validation label: {}".format(val_label.shape))
print()
#%%
'''
    model building parts
'''
class_num = 2
patch_size = 48
s1,s2,s3 = patch_size, patch_size, patch_size
images = tf.placeholder(tf.float32, (None, s1 * 2, s2, s3, 1), name='inputs')
lh, rh = tf.split(images, [patch_size, patch_size], 1)
y_gt = tf.placeholder(tf.float32, (None, 2))
keep_prob = tf.placeholder(tf.float32)
x = batch_norm(lh)
x = tf.layers.conv3d(inputs=x, filters=32, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)
x = tf.layers.conv3d(inputs=x, filters=64, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)
x = tf.layers.conv3d(inputs=x, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)
x = tf.layers.conv3d(inputs=x, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
x = tf.layers.flatten(x)
# x = tf.layers.dense(x, units=2048, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, use_bias=use_bias)
x = tf.layers.dense(x, units=2048, activation=tf.nn.relu)
x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
x = tf.layers.dense(x, units=class_num, activation=tf.nn.sigmoid)
y = x

# layer_1_w = tf.Variable(tf.truncated_normal(shape=(5,5,3,32), mean=0, stddev=0.1))
# layer_1_b = tf.Variable(tf.zeros(32))
# layer_1 = tf.nn.conv2d(x, layer_1_w, strides=[1,1,1,1], padding='SAME') + layer_1_b
# layer_1 = tf.nn.relu(layer_1)
# layer_1 = tf.nn.max_pool(layer_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#
# layer_2_w = tf.Variable(tf.truncated_normal(shape=(5,5,32,64), mean=0, stddev=0.1))
# layer_2_b = tf.Variable(tf.zeros(64))
# layer_2 = tf.nn.conv2d(layer_1, layer_2_w, strides=[1,1,1,1], padding='SAME') + layer_2_b
# layer_2 = tf.nn.relu(layer_2)
# layer_2 = tf.nn.max_pool(layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#
# flat_arr = tf.reshape(layer_2, [-1, int(img_size[0]/4)*int(img_size[1]/4)*64])
#
# fcl_1_w = tf.Variable(tf.truncated_normal(shape=(int(img_size[0]/4)* \
#     int(img_size[1]/4)*64,1024), mean=0, stddev=0.1))
# fcl_1_b = tf.Variable(tf.zeros(1024))
# fcl_1 = tf.matmul(flat_arr, fcl_1_w) + fcl_1_b
# fcl_1 = tf.nn.dropout(fcl_1, keep_prob)
#
# fcl_2_w = tf.Variable(tf.truncated_normal(shape=(1024,2), mean=0, stddev=0.1))
# fcl_2_b = tf.Variable(tf.zeros(2))
# fcl_2 = tf.matmul(fcl_1, fcl_2_w) + fcl_2_b
#
# y = fcl_2


#%%

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_gt, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

saver = tf.train.Saver(max_to_keep = 0)
init = tf.global_variables_initializer()

#%%
batch = 50
dropout_prob = 0.5
epochs = 200
epoch_freq = 1

data_read_test = False
if data_read_test:
    print("test data reading ... not training ...")
    next_element, iterator = get_patch_dataset(train_data, train_label, batch)
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(3):
            train_data, train_label = sess.run(next_element)
            # print(train_label)
            print(train_data)
    assert False

with tf.Session() as sess:
    try:
        saver.restore(sess, "../train/dogs_cats_cnn")
        print()
        print('Initialization loaded')
        print()
    except:
        sess.run(init)
        print()
        print('New initialization done')
        print()

    # tensorflow dataset setting
    next_element, iterator = get_patch_dataset(train_data, train_label, batch)
    sess.run(iterator.initializer)

    for epoch in range(epochs):
        train_data, train_label = sess.run(next_element)
        # train_data, train_label = shuffle(train_data, train_label)
        accum_loss = 0
        accum_acc = 0

        _, loss_scr, acc_scr, logit = sess.run((train_step, loss, accuracy, y), \
            feed_dict = {images: train_data, y_gt: train_label, keep_prob: dropout_prob})

        accum_loss += loss_scr
        accum_acc += acc_scr

        if epoch%epoch_freq == 0:
            print("Epoch: {}".format(epoch))
            print("Train loss = {}".format(accum_loss/train_data.shape[0]))
            print("Train accuracy = {:03.4f}".format(accum_acc/train_data.shape[0]))
            print(logit[:5])

            # accum_acc = 0
            #
            # for m in range(0, val_data.shape[0], batch):
            #     m2 = min(val_data.shape[0], m+batch)
            #
            #     acc_scr = sess.run((accuracy), \
            #         feed_dict = {x: val_data[m:m2], y_gt: val_label[m:m2], \
            #         keep_prob: 1})
            #
            #     accum_acc += acc_scr*(m2-m)
            # print("Validation accuracy = {:03.4f}".format(accum_acc/val_data.shape[0]))
            print()

        save_path = saver.save(sess, "../train/dogs_cats_cnn")


#%%

print("This is the end of the training")
print("Entering in testing mode")
print()
assert False

img_size = (28,28)

test_data, test_label = load_data(img_folder, test_list, img_size)

print("test data: {}".format(test_data.shape))
print("test label: {}".format(test_label.shape))
print()


#%%

with tf.Session() as sess:

    saver.restore(sess, "./train/dogs_cats_cnn")
    print()
    print('Initialization loaded')
    print()

    accum_acc = 0

    for m in range(0, test_data.shape[0], batch):
        m2 = min(test_data.shape[0], m+batch)
        
        acc_scr = sess.run((accuracy), \
            feed_dict = {x: test_data[m:m2], y_gt: test_label[m:m2], \
            keep_prob: 1})

        accum_acc += acc_scr*(m2-m)
    print("Test accuracy = {:03.4f}".format(accum_acc/test_data.shape[0]))
    print()
