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
    parser.add_argument('--epoch',              default=50, type=int)
    parser.add_argument('--network',            default='simple', type=str) # simple attention siam
    parser.add_argument('--lr',                 default=1e-5, type=float)
    parser.add_argument('--ch',                 default=32, type=int)
    parser.add_argument('--fold',           default=1, type=int)
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

def what_time():
    now = time.gmtime(time.time())
    now_list = [str(now.tm_year) , str(now.tm_mon ), str(now.tm_mday ), str(now.tm_hour ), str(now.tm_min ), str(now.tm_sec)]
    return ''.join(now_list)

def test():
    a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    b = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    report = classification_report(a, b, target_names=['NC', 'AD'])
    report = report.split(' ')
    for e in reversed(report):
        if e == '':
            report.remove('')
    # print(report)
    # report.remove('')
    print(report)
    assert False

if __name__ == '__main__':
    # test()

    ch = args.ch
    batch = args.batch_size # 10
    dropout_prob = 0.5
    epochs = args.epoch
    is_mask = args.mask
    print_freq = 1
    learning_rate = args.lr

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
    elif args.network == 'attention2':
        network = Attention2
    else:
        assert False
    assert network != None

    # patch_num = 2
    initializer = tf.contrib.layers.xavier_initializer()
    my_model = network(weight_initializer=initializer,
                      activation=tf.nn.relu,
                      class_num=class_num,
                      patch_size=s1,
                      patch_num=patch_num)
    y = my_model.model(images)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y)
    loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('learning_rate_decay'):
        start_lr = learning_rate
        global_step = tf.Variable(0, trainable=False)
        total_learning = epochs
        lr = learning_rate

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
    count = 0
    check_position = 0
    min_val_loss = 100

    fold = args.fold
    class_num = 2
    sampling_option = "SIMPLE"
    train_data, train_label, val_data, val_label = whole_set[fold]
    val_data, val_label = valence_class(val_data, val_label, class_num)

    data_count = len(train_label)
    print("validation data: {}".format(val_data.shape))
    print("validation label: {}".format(val_label.shape))
    print()

    # model_vars = tf.trainable_variables()
    # tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    # --------------------- tensorflow dataset setting --------------------- #
    # test_element, test_iterator = get_patch_dataset(val_data, val_label, args.buffer_scale, is_mask, len(val_label))
    # sess.run(test_iterator.initializer)
    # val_data_ts, test_label_ts = sess.run(test_element)
    val_data_ts, _ = read_test_data(val_data, val_label, is_masking=is_mask)
    test_label_ts = one_hot_pd(val_label)

    # --------------------- network construction  --------------------- #
    ckpt_state = tf.train.get_checkpoint_state('../checkpoint')
    recent_ckpt_job_path = tf.train.latest_checkpoint("saved")

    print(ckpt_state.model_checkpoint_path)
    print(ckpt_state.all_model_checkpoint_paths)
    print(recent_ckpt_job_path)
    # assert False
    saver = tf.train.import_meta_graph('../checkpoint/model.meta')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint('../checkpoint'))
        saver.restore(sess, '../checkpoint/model')
        # print(sess.run(''))
        # assert False
        # sess.run(init)
        print()
        print('testing ... ')
        print('<< Try fold {} .. >>'.format(fold))
        print()
        x = tf.get_collection('images')[0]
        y = tf.get_collection('y_gt')[0]
        test_feed_dict = {
            x: val_data_ts,
            y_gt: test_label_ts
        }

        test_writer = tf.summary.FileWriter('../log/test/'+args.network+what_time(), sess.graph)
        val_acc, val_logit, val_loss, test_summary = \
            sess.run((accuracy, y, loss, merged_summary), feed_dict=test_feed_dict)

        pn = 4
        print(val_logit[:pn]//0.01)
        # print(val_logit[:pn]//0.01)
        test_writer.add_summary(test_summary)
        target = list(val_label)
        pred = list(np.argmax(val_logit, axis=1))
        report = classification_report(target, pred, target_names=['NC', 'AD'])
        print(report)