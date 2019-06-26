import os
import sys
import time
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
sys.path.append('.')
sys.path.append('..')
from data import *
from data_merge import *
from model import *

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
    parser.add_argument('--epoch',              default=40, type=int)
    parser.add_argument('--network',            default='simple', type=str) # simple attention siam
    parser.add_argument('--lr',                 default=1e-5, type=float)
    parser.add_argument('--ch',                 default=32, type=int)
    parser.add_argument('--fold',           default=1, type=int)
    parser.add_argument('--batch_size',         default=10, type=int)
    return parser.parse_args()

def what_time():
    now = time.gmtime(time.time())
    now_list = [str(now.tm_year) , str(now.tm_mon ), str(now.tm_mday ), str(now.tm_hour ), str(now.tm_min ), str(now.tm_sec)]
    return ''.join(now_list)

def classifier(sv_set, args):
    # ----------------- data load part ---------------- #
    whole_set = read_cnn_data(sv_set)
    tr_x, tr_y, tst_x, tst_y = whole_set[args.fold]
    tr_x, val_x, tr_y, val_y = train_test_split(tr_x, tr_y, test_size = 0.1, random_state = 42)
    tr_x, tr_y = over_sampling(tr_x, tr_y, "SIMPLE")
    print(len(tr_y), len(val_y), len(tst_y))
    # ----------------- graph build part ---------------- #
    batch_size = 10
    buff_size = 1000
    img_size = 192
    class_num = 2
    next_element, iterator, iters = define_dataset(tr_x, tr_y, batch_size, buffer_size=buff_size)
    x = tf.reshape(next_element[0], shape=[batch_size,img_size,img_size,img_size,1])
    y_gt = tf.one_hot(next_element[1], class_num)
    print(x, y_gt)

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

    # weight initialzier
    initializer = tf.contrib.layers.xavier_initializer()
    # initializer = tf.truncated_normal_initializer
    net = network(weight_initializer=initializer,
                       activation=tf.nn.relu,
                       class_num=class_num,
                       patch_size=1,
                       patch_num=1)
    y_pred = net.model(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_gt, logits=y_pred)
    # L2 = tf.nn.l2_loss(x,y_pr)
    loss = tf.reduce_mean(cross_entropy)
    with tf.variable_scope('optimizer'):
        rate = tf.placeholder(dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(rate)
        train_step = optimizer.minimize(loss)
        # print(y_pred.shape)
        # print(y_pred)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_gt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summarize
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    epochs = 1000
    lr = 0.001
    for epoch in range(epochs):
        for iter in range(iters):
            _, loss_tr, accur_tr = sess.run((train_step, loss, accuracy),
                                            feed_dict={rate:lr})
            print("epoch : {}/{} - iter: {}/{} - train loss : {:02.4} - train accur : {:02.3}"
                  .format(epoch, epochs, iter, iters, loss_tr, accur_tr // 0.01))

        print(epoch,loss_tr, accur_tr)

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sv_set_dict = {
        "desktop": 0,
        "sv186": 186,
        "sv144": 144,
        "sv202": 202,
    }
    sv_set = sv_set_dict[args.setting]
    classifier(sv_set, args)