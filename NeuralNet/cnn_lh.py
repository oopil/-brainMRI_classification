import tensorflow as tf
import os
import sys
import argparse
from sklearn.model_selection import KFold

# from NeuralNet.CNN_data import *
# from NeuralNet.NN_ops import *

sys.path.append('..')
sys.path.append('/home/soopil/Desktop/Dataset/github/brainMRI_classification/NeuralNet')
from data_merge import *

# server setting
from CNN_data import *
from NN_ops import *

# %%
def parse_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--setting', default='desktop', type=str)
    parser.add_argument('--mask', default=True, type=bool)
    parser.add_argument('--buffer_scale', default=3, type=int)
    return parser.parse_args()

# %%
args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
sv_set_dict = {
    "desktop":0,
    "sv186":186,
    "sv144":144,
    "sv202":202,
}
sv_set = sv_set_dict[args.setting]
# %%

def read_cnn_data(sv_set = 0):
    if sv_set == 186:
        base_folder_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_empty_copy'
        excel_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 0:
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

    train_data, train_label, val_data, val_label = whole_set[0]
    val_data, val_label = valence_class(val_data, val_label, class_num)
    if sampling_option != "None":
        train_data, train_label = over_sampling(train_data, train_label, sampling_option)
    return train_data, one_hot_pd(train_label), val_data, one_hot_pd(val_label)
    # print(train_data)
    # print(type(train_data))
train_data, train_label, val_data, val_label = read_cnn_data()

'''
    model building parts
'''
class_num = 2
patch_size = 48
s1, s2, s3 = patch_size, patch_size, patch_size
images = tf.placeholder(tf.float32, (None, s1 * 2, s2, s3, 1), name='inputs')
# a = batch_norm(images)
lh, rh = tf.split(images, [patch_size, patch_size], 1)
y_gt = tf.placeholder(tf.float32, (None, 2))
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("Model"):
    with tf.variable_scope("Left"):
        lh = batch_norm(lh)
        lh = tf.layers.conv3d(inputs=lh, filters=32, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
        lh = tf.layers.conv3d(inputs=lh, filters=64, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
        lh = tf.layers.conv3d(inputs=lh, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.max_pooling3d(inputs=lh, pool_size=[2, 2, 2], strides=2)
        lh = tf.layers.conv3d(inputs=lh, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        lh = tf.layers.flatten(lh)

    with tf.variable_scope("FCN"):
        # x = tf.concat([lh,rh], -1)
        x = lh
        x = tf.layers.dense(x, units=2048, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=class_num, activation=tf.nn.sigmoid)
        y = x
# %%
is_mask = True
buffer_scale = args.buffer_scale
batch = 30
dropout_prob = 0.5
epochs = 100
epoch_freq = 1
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
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Summarize
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

saver = tf.train.Saver(max_to_keep=0)
init = tf.global_variables_initializer()

# %%

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
        saver.restore(sess, "../train/cnn_lh")
        print()
        print('Initialization loaded')
        print()
    except:
        sess.run(init)
        print()
        print('New initialization done')
        print()

    # tensorflow dataset setting
    next_element, iterator = get_patch_dataset(train_data, train_label, buffer_scale, is_mask, batch)
    sess.run(iterator.initializer)
    test_element, test_iterator = get_patch_dataset(val_data, val_label, buffer_scale, is_mask, len(val_label))
    sess.run(test_iterator.initializer)
    val_data_ts, test_label_ts = sess.run(test_element)

    train_writer = tf.summary.FileWriter('../log/train', sess.graph)
    test_writer = tf.summary.FileWriter('../log/test')
    for epoch in range(epochs):
        train_data, train_label = sess.run(next_element)
        train_feed_dict = {
            images: train_data,
            y_gt: train_label
        }
        test_feed_dict = {
            images: val_data_ts,
            y_gt: test_label_ts
        }
        # train_data, train_label = shuffle(train_data, train_label)
        accum_loss = 0
        accum_acc = 0

        _, loss_scr, acc_scr, logit, train_summary = sess.run((train_step, loss, accuracy, y, merged_summary), feed_dict=train_feed_dict)
        train_writer.add_summary(train_summary)
        accum_loss += loss_scr
        accum_acc += acc_scr

        if epoch % epoch_freq == 0:
            print("Epoch: {}".format(epoch))
            print("Train loss = {}".format(loss_scr))
            print("Train accuracy = {:03.4f}".format(acc_scr // 0.01))
            val_acc, val_logit, test_summary = sess.run((accuracy,y, merged_summary), feed_dict=test_feed_dict)
            print("Validation accuracy = {:03.4f}".format(val_acc // 0.01))
            print(logit[:5])
            print(val_logit[:5])
            train_writer.add_summary(test_summary)

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

    # save trained model
    # save_path = saver.save(sess, "../train/cnn_lh")
# %%

print("This is the end of the training")

whole_set = read_cnn_data(sv_set)
train_result = []
valid_result = []
for fold in whole_set:
    class_num = 2
    sampling_option = "RANDOM"

    train_data, train_label, val_data, val_label = fold
    val_data, val_label = valence_class(val_data, val_label, class_num)
    if sampling_option != "None":
        train_data, train_label = over_sampling(train_data, train_label, sampling_option)
        train_label, val_label = one_hot_pd(train_label), one_hot_pd(val_label)

    print()
    print("Loading data...")
    print()
    print()
    print("train data: {}".format(train_data.shape))
    print("train label: {}".format(train_label.shape))
    print()
    print("validation data: {}".format(val_data.shape))
    print("validation label: {}".format(val_label.shape))
    print()
    tr, val = cross_validate(train_data, train_label, val_data, val_label, is_masking = args.mask, args=args)
    train_result.append(tr)
    valid_result.append(val)

for i in range(len(train_result)):
    print("<< fold {} result>>".format(i))
    print("CNN lh model")
    print("masking : {}".format(args.mask))
    print("train : {}".format(train_result))
    print("valid : {}".format(valid_result))

# %%

print("This is the end of the training")
"""
print("Entering in testing mode")
print()
img_size = (28, 28)

val_data, val_label = load_data(img_folder, test_list, img_size)

print("test data: {}".format(val_data.shape))
print("test label: {}".format(val_label.shape))
print()

# %%

with tf.Session() as sess:
    saver.restore(sess, "./train/dogs_cats_cnn")
    print()
    print('Initialization loaded')
    print()

    accum_acc = 0

    for m in range(0, val_data.shape[0], batch):
        m2 = min(val_data.shape[0], m + batch)

        acc_scr = sess.run((accuracy), \
                           feed_dict={x: val_data[m:m2], y_gt: val_label[m:m2], \
                                      keep_prob: 1})

        accum_acc += acc_scr * (m2 - m)
    print("Test accuracy = {:03.4f}".format(accum_acc / val_data.shape[0]))
    print()

"""
