import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
# from ConvNeuralNet.CNN_net import *

sys.path.append('..')
sys.path.append('/home/soopil/Desktop/Dataset/github/brainMRI_classification/ConvNeuralNet')
from data_merge import *

# server setting
from CNN_data import *
from CNN_ops import *
from CNN_net import *

# %%
def parse_args() -> argparse:
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',                default='0', type=str)
    parser.add_argument('--setting',            default='desktop', type=str) # desktop sv186 sv202 sv144
    parser.add_argument('--mask',               default=False, type=str2bool)
    parser.add_argument('--buffer_scale',       default=30, type=int)
    parser.add_argument('--epoch',              default=100, type=int)
    parser.add_argument('--network',            default='simple', type=str) # simple attention siam
    parser.add_argument('--lr',                 default=1e-5, type=float) # simple attention siam
    parser.add_argument('--ch',                 default=32, type=int) # simple attention siam
    parser.add_argument('--fold_try',           default=2, type=int)
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

def read_cnn_data(sv_set = 0):
    base_folder_path = ''
    excel_path = ''
    if sv_set == 186:
        base_folder_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_empty_copy'
        excel_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 0: # desktop
        base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 202:
        base_folder_path = '/home/soopil/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        excel_path = '/home/soopil/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    else:
        assert False

    diag_type = 'clinic'
    class_option = 'CN vs AD'
    class_num = 2
    test_num = 20
    fold_num = 5
    is_split_by_num = False
    sampling_option = "RANDOM"
    whole_set = CNN_dataloader(base_folder_path, diag_type, class_option, excel_path, fold_num)
    return whole_set
    # print(train_data)
    # print(type(train_data))


ch = args.ch
batch = 30
dropout_prob = 0.5
epochs = args.epoch
is_mask = args.mask
print_freq = 5
learning_rate = args.lr
'''
    model building parts
'''
class_num = 2
patch_size = 48
s1, s2, s3 = patch_size, patch_size, patch_size
images = tf.placeholder(tf.float32, (None, s1 * 2, s2, s3, 1), name='inputs')

lh, rh = tf.split(images, [patch_size, patch_size], 1)
y_gt = tf.placeholder(tf.float32, (None, 2))
keep_prob = tf.placeholder(tf.float32)

# network = None
# if args.network == 'simple':
#     network = Simple
# elif args.network == 'siam':
#     network = Siamese
# else:
#     assert False
# assert network != None
#
# my_model = network(weight_initializer=tf.truncated_normal_initializer,
#                   activation=tf.nn.relu,
#                   class_num=class_num,
#                   patch_size=s1,
#                   patch_num=2)
def model(images):
    def CNN_simple( x, ch = 32, scope = "CNN", reuse = False):
        activ = tf.nn.relu
        ps = 48
        cn = 2

        with tf.variable_scope(scope, reuse=reuse):
            x = batch_norm(x)
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            x = maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            x = maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            x = maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            x = conv_3d(x, ch, [3, 3, 3], 'same', activ)
            return x
    is_print = False
    # is_print = is_print
    if is_print:
        print('build neural network')
        print(images.shape)

    with tf.variable_scope("Model"):
        # images = tf.placeholder(tf.float32, (None, ps * 2, ps, ps, 1), name='inputs')
        lh, rh = tf.split(images, [ps, ps], 1)
        # output_num = 3
        # tf.summary.image('lh_orig_1',lh[0],max_outputs=output_num)
        # tf.summary.image('lh_orig_2',lh[0],max_outputs=output_num)
        # tf.summary.image('rh_orig',rh[0],max_outputs=output_num)
        flip_axis = 3  # 3
        # axis = [False for i in range(5)]
        # axis[flip_axis] = True
        rh = tf.reverse(rh, axis=[flip_axis])
        # tf.summary.image('rh',rh[0],max_outputs=output_num)
        # CNN = CNN_deep_layer
        CNN = CNN_simple

        # channel = 32
        channel = 40
        lh = CNN(lh, ch=channel, scope="CNN", reuse=False)
        rh = CNN(rh, ch=channel, scope="CNN", reuse=True)

        with tf.variable_scope("FCN"):
            lh = tf.layers.flatten(lh)
            rh = tf.layers.flatten(rh)
            x = tf.concat([lh, rh], -1)
            # x = tf.subtract(lh,rh)
            x = tf.layers.dense(x, units=2048, activation=activ)
            x = tf.layers.dense(x, units=512, activation=activ)
            x = tf.layers.dense(x, units=cn, activation=tf.nn.softmax)
            # x = tf.layers.dense(x, units=cn, activation=tf.nn.sigmoid)
            y = x
    return y

y = my_model.model(images)
# %%
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y)
loss = tf.reduce_mean(cross_entropy)

# with tf.name_scope('learning_rate_decay'):
#     start_lr = learning_rate
#     global_step = tf.Variable(0, trainable=False)
#     total_learning = epochs
#     lr = tf.train.exponential_decay(start_lr, global_step, total_learning, 0.99999, staircase=True)
with tf.name_scope('learning_rate_decay'):
    start_lr = learning_rate
    global_step = tf.Variable(0, trainable=False)
    total_learning = epochs
    # lr = tf.train.exponential_decay(start_lr, global_step,total_learning,0.99999, staircase=True)
    lr = tf.train.exponential_decay(start_lr, global_step, decay_steps=epochs // 100, decay_rate=.96, staircase=True)

with tf.variable_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_gt, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Summarize
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

whole_set = read_cnn_data(sv_set)
top_train_accur_list = []
top_valid_accur_list = []
saturation_train_accur_list = []
saturation_valid_accur_list = []
train_result = []
valid_result = []
train_accur = []
valid_accur = []
count = 0
for fold in whole_set:
    acc_scr, val_acc = 0,0
    train_accur = []
    valid_accur = []
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
    # saver = tf.train.Saver(max_to_keep=0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print()
        print('Cross Validation Step ... ')
        print()
        # tensorflow dataset setting
        next_element, iterator = get_patch_dataset(train_data, train_label, args.buffer_scale, is_mask, batch)
        sess.run(iterator.initializer)
        test_element, test_iterator = get_patch_dataset(val_data, val_label, args.buffer_scale, is_mask, len(val_label))
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
            accum_loss = 0
            accum_acc = 0

            _, loss_scr, acc_scr, logit, train_summary = \
                sess.run((train_step, loss, accuracy, y, merged_summary), feed_dict=train_feed_dict)

            train_writer.add_summary(train_summary)
            if epoch % print_freq == 0:
                print("Epoch: {}".format(epoch))
                print("Train loss = {}".format(loss_scr))
                print("Train accuracy = {:03.4f}".format(acc_scr // 0.01))

                val_acc, val_logit, test_summary = \
                    sess.run((accuracy, y, merged_summary), feed_dict=test_feed_dict)

                print("Validation accuracy = {:03.4f}".format(val_acc // 0.01))
                print(logit[:5])
                # print(val_logit[:5])
                train_writer.add_summary(test_summary)
                train_accur.append(acc_scr)
                valid_accur.append(val_acc)
                # save trained model
                # save_path = saver.save(sess, "../train/cnn_lh")
    count += 1
    if count >= args.fold_try:
        break

    saturation_count = 5
    train_result.append(train_accur)
    valid_result.append(valid_accur)
    top_train_accur = np.max(train_accur, 0)
    top_valid_accur = np.max(valid_accur, 0)
    top_train_accur_list.append(top_train_accur)
    top_valid_accur_list.append(top_valid_accur)
    saturation_train_accur_list.append(np.mean(train_accur[-saturation_count:]))
    saturation_valid_accur_list.append(np.mean(valid_accur[-saturation_count:]))

file_contents = []

for i in range(len(train_result)):
    file_contents.append("<< fold {} result>>".format(i))
    file_contents.append("CNN lh and rh model")
    file_contents.append("masking : {}".format(args.mask))
    file_contents.append("train : {}".format(train_result[i]))
    file_contents.append("valid : {}".format(valid_result[i]))
file_contents.append("top train : {}".format(top_train_accur_list))
file_contents.append("top valid : {}".format(top_valid_accur_list))
file_contents.append("avg train top : {} , avg vaidation top : {}".format(np.mean(top_train_accur_list), np.mean(top_valid_accur_list)))
file_contents.append("saturation train : {}".format(saturation_train_accur_list))
file_contents.append("saturation valid : {}".format(saturation_valid_accur_list))
file_contents.append("avg saturation train : {} , avg saturation vaidation : {}".format(np.mean(saturation_train_accur_list), np.mean(saturation_valid_accur_list)))

for result in file_contents:
    print(result)

result_file_name = '../nn_result_'+args.network+'/cv.txt'
file = open(result_file_name, 'a+t')
for result in file_contents:
    result += '\n'
    file.writelines(result)

"""

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

    with tf.variable_scope("Right", reuse=False):
        rh = batch_norm(rh)
        rh = tf.layers.conv3d(inputs=rh, filters=32, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
        rh = tf.layers.conv3d(inputs=rh, filters=64, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
        rh = tf.layers.conv3d(inputs=rh, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.max_pooling3d(inputs=rh, pool_size=[2, 2, 2], strides=2)
        rh = tf.layers.conv3d(inputs=rh, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        rh = tf.layers.flatten(rh)

    with tf.variable_scope("FCN"):
        x = tf.concat([lh, rh], -1)
        x = tf.layers.dense(x, units=2048, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=class_num, activation=tf.nn.sigmoid)
        y = x
"""