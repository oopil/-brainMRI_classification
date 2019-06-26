import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
# report = classification_report(test_y, arg_ytest, target_names=['low','high'])
# print(report)

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
    parser.add_argument('--epoch',              default=400, type=int)
    parser.add_argument('--network',            default='simple', type=str) # simple attention siam
    parser.add_argument('--lr',                 default=1e-5, type=float)
    parser.add_argument('--ch',                 default=32, type=int)
    parser.add_argument('--fold_try',           default=1, type=int)
    parser.add_argument('--fold_start',           default=1, type=int)
    parser.add_argument('--batch_size',         default=10, type=int)
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
# is_mask = args.mask
# print(is_mask)
# print(type(is_mask))
# assert False
# %%

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
    elif sv_set == 144:
        base_folder_path = '/user/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        excel_path = '/user/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    else:
        assert False

    diag_type = 'clinic'
    class_option = 'CN vs AD'
    fold_num = 5
    whole_set = CNN_dataloader(base_folder_path, diag_type, class_option, excel_path, fold_num)
    return whole_set
    # print(train_data)
    # print(type(train_data))

def what_time():
    now = time.gmtime(time.time())
    now_list = [str(now.tm_year) , str(now.tm_mon ), str(now.tm_mday ), str(now.tm_hour ), str(now.tm_min ), str(now.tm_sec)]
    return ''.join(now_list)

ch = args.ch
batch = args.batch_size # 10
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
# patch_size = 16
patch_num = 2 # hippocampus labels
# patch_num = 70 # cortical labels
# patch_num = 2 + 32 + 32 # hippo + cortical labels
# patch_num = 34 #2 + 32 + 32 # hippo + cortical labels
# patch_num = 20 # subcortical labels
s1, s2, s3 = patch_size, patch_size, patch_size
images = tf.placeholder(tf.float32, (None, s1 * patch_num, s2, s3, 1), name='inputs')
# lh, rh = tf.split(images, [patch_size, patch_size], 1)
y_gt = tf.placeholder(tf.float32, (None, 2))
keep_prob = tf.placeholder(tf.float32)

network = None
if args.network == 'simple':
    network = Simple
elif args.network == 'siam':
    network = Siamese
elif args.network == 'attention':
    network = Attention
else:
    assert False
assert network != None

# patch_num = 2
initializer = tf.contrib.layers.xavier_initializer()
# initializer = tf.truncated_normal_initializer

my_model = network(weight_initializer=initializer,
                  activation=tf.nn.relu,
                  class_num=class_num,
                  patch_size=s1,
                  patch_num=patch_num)
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

# model_vars = tf.trainable_variables()
# tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

whole_set = read_cnn_data(sv_set)
top_train_accur_list = []
top_valid_accur_list = []
top_valid_index_list = []
saturation_train_accur_list = []
saturation_valid_accur_list = []
train_result = []
valid_result = []
train_accur = []
valid_accur = []
count = 0
check_position = 0
min_val_loss = 10
for fold in whole_set:
    if count < args.fold_start:
        count += 1
        continue
    acc_scr, val_acc = 0,0
    train_accur = []
    valid_accur = []
    class_num = 2
    # sampling_option = "None"
    # sampling_option = "RANDOM"
    sampling_option = "SIMPLE"
    train_data, train_label, val_data, val_label = fold
    val_data, val_label = valence_class(val_data, val_label, class_num)
    if sampling_option != "None":
        train_data, train_label = over_sampling(train_data, train_label, sampling_option)
        train_label = one_hot_pd(train_label)

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

    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print()
        print('Cross Validation Step ... ')
        print('<< Try fold {} .. >>'.format(count))
        print()
        # tensorflow dataset setting
        next_element, iterator = get_patch_dataset(train_data, train_label, args.buffer_scale, is_mask, batch)
        sess.run(iterator.initializer)
        val_data_ts, test_label_ts = read_test_data(val_data, val_label, is_masking=is_mask)
        test_label_ts = one_hot_pd(val_label)
        print(test_label_ts.shape)

        # train_writer = tf.summary.FileWriter('../log/train/'+what_time(), sess.graph)
        test_writer = tf.summary.FileWriter('../log/test/'+what_time())

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

            # train_writer.add_summary(train_summary, global_step=epoch)
            if epoch % print_freq == 0:
                val_acc, val_logit, val_loss, test_summary = \
                    sess.run((accuracy, y, loss, merged_summary), feed_dict=test_feed_dict)
                print("Epoch: {}/{} - train loss : {:02.4} - train accur : {:02.3} - val loss : {:02.4} - val accur : {:02.3}".format(epoch, epochs, loss_scr, acc_scr // 0.01, val_loss, val_acc // 0.01))
                pn = 4
                print(logit[:pn]//0.01)
                # print(val_logit[:pn]//0.01)
                # train_writer.add_summary(test_summary)
                train_accur.append(acc_scr)
                valid_accur.append(val_acc)

                if val_loss < min_val_loss and epoch > 200:
                    min_val_loss = val_loss
                    check_position = epoch
                    print('save the checkpoint ... ', epoch)
                    # save trained model
                    save_path = saver.save(sess, "../checkpoint/model")
        print('last check point epoch : ' ,check_position)
    saturation_count = 5
    train_result.append(train_accur)
    valid_result.append(valid_accur)
    top_train_accur = np.max(train_accur, 0)
    top_valid_accur = np.max(valid_accur, 0)
    top_valid_index = np.where(valid_accur == top_valid_accur)
    top_train_accur_list.append(top_train_accur)
    top_valid_accur_list.append(top_valid_accur)
    top_valid_index_list.append(top_valid_index)
    saturation_train_accur_list.append(np.mean(train_accur[-saturation_count:]))
    saturation_valid_accur_list.append(np.mean(valid_accur[-saturation_count:]))
    count += 1
    if count >= args.fold_try:
        break
file_contents = []