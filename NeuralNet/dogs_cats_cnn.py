import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import os
import sys
sys.path.append('..')
import pandas as pd
# from NeuralNet.CNN_data import *
# from NeuralNet.NN_ops import *
from sklearn.utils import shuffle
from data_merge import *
# server setting
from CNN_data import *
from NN_ops import *


#%%

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%%


def read_cnn_data():
    # base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
    base_folder_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_empty_copy' # sv186
    # excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    excel_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_test.xlsx' # sv186
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
with tf.variable_scope("Left", reuse=False):
    lh = batch_norm(lh)
    lh = tf.layers.conv3d(inputs=lh, filters=32, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
    lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
    lh = tf.layers.conv3d(inputs=lh, filters=64, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
    lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
    lh = tf.layers.conv3d(inputs=lh, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
    lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
    lh = tf.layers.conv3d(inputs=lh, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
    lh = tf.layers.flatten(lh)

# with tf.variable_scope("Right", reuse=False):
#     rh = batch_norm(rh)
#     rh = tf.layers.conv3d(inputs=rh, filters=32, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
#     rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
#     rh = tf.layers.conv3d(inputs=rh, filters=64, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
#     rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
#     rh = tf.layers.conv3d(inputs=rh, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
#     rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
#     rh = tf.layers.conv3d(inputs=rh, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
#     rh = tf.layers.flatten(rh)

# x = tf.layers.dense(x, units=2048, kernel_initializer=weight_initializer, kernel_regularizer=weight_regularizer, use_bias=use_bias)
# x = tf.concat([lh,rh], -1)
x = lh
x = tf.layers.dense(x, units=2048, activation=tf.nn.relu)
x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
x = tf.layers.dense(x, units=class_num, activation=tf.nn.sigmoid)
y = x
#%%
batch = 30
dropout_prob = 0.5
epochs = 20000
epoch_freq = 10
learning_rate = 1e-4
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y)
loss = tf.reduce_mean(cross_entropy)

with tf.name_scope('learning_rate_decay'):
    start_lr = learning_rate
    global_step = tf.Variable(0, trainable=False)
    total_learning = epochs
    lr = tf.train.exponential_decay(start_lr, global_step, total_learning, 0.99999, staircase=True)
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_gt, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

saver = tf.train.Saver(max_to_keep = 0)
init = tf.global_variables_initializer()

#%%

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
