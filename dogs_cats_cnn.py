import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import os
import cv2
from sklearn.utils import shuffle


#%%

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%%

img_folder = './tiny_train'
# img_folder = '/user/Datasets/dogs_and_cats/train'

img_list = os.listdir(img_folder)
img_list.sort()
number_of_imgs = len(img_list)


#%% 

train_list = img_list[:int(number_of_imgs*0.25)] + \
    img_list[int(number_of_imgs*0.5):int(number_of_imgs*0.75)]

val_list = img_list[int(number_of_imgs*0.25):int(number_of_imgs*0.3)] + \
    img_list[int(number_of_imgs*0.75):int(number_of_imgs*0.8)]

test_list = img_list[int(number_of_imgs*0.3):int(number_of_imgs*0.5)] + \
    img_list[int(number_of_imgs*0.8):]


#%%

def load_data(folder, files_list, size):

    data_list = []
    label_list = []
    for img_name in files_list:
        img = mpimg.imread(folder + '/' + img_name)
        if img.shape[2] == 3:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            img_min = np.min(img)
            img_max = np.max(img)
            img = (img-img_min)/(img_max-img_min)
            data_list.append(img)
            if img_name[:3] == 'cat':
                label_list.append([1,0])
            else:
                label_list.append([0,1])

    data = np.array(data_list)
    labels = np.array(label_list)

    return data, labels


#%%

print()
print("Loading data...")
print()

img_size = (28,28)

train_data, train_label = load_data(img_folder, train_list, img_size)
val_data, val_label = load_data(img_folder, val_list, img_size)

print()
print("train data: {}".format(train_data.shape))
print("train label: {}".format(train_label.shape))
print()
print("validation data: {}".format(val_data.shape))
print("validation label: {}".format(val_label.shape))
print()


#%%

x = tf.placeholder(tf.float32, (None, img_size[0], img_size[1], 3))
y_gt = tf.placeholder(tf.float32, (None, 2))
keep_prob = tf.placeholder(tf.float32)


layer_1_w = tf.Variable(tf.truncated_normal(shape=(5,5,3,32), mean=0, stddev=0.1))
layer_1_b = tf.Variable(tf.zeros(32))
layer_1 = tf.nn.conv2d(x, layer_1_w, strides=[1,1,1,1], padding='SAME') + layer_1_b
layer_1 = tf.nn.relu(layer_1)
layer_1 = tf.nn.max_pool(layer_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

layer_2_w = tf.Variable(tf.truncated_normal(shape=(5,5,32,64), mean=0, stddev=0.1))
layer_2_b = tf.Variable(tf.zeros(64))
layer_2 = tf.nn.conv2d(layer_1, layer_2_w, strides=[1,1,1,1], padding='SAME') + layer_2_b
layer_2 = tf.nn.relu(layer_2)
layer_2 = tf.nn.max_pool(layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

flat_arr = tf.reshape(layer_2, [-1, int(img_size[0]/4)*int(img_size[1]/4)*64])

fcl_1_w = tf.Variable(tf.truncated_normal(shape=(int(img_size[0]/4)* \
    int(img_size[1]/4)*64,1024), mean=0, stddev=0.1))
fcl_1_b = tf.Variable(tf.zeros(1024))
fcl_1 = tf.matmul(flat_arr, fcl_1_w) + fcl_1_b
fcl_1 = tf.nn.dropout(fcl_1, keep_prob)

fcl_2_w = tf.Variable(tf.truncated_normal(shape=(1024,2), mean=0, stddev=0.1))
fcl_2_b = tf.Variable(tf.zeros(2))
fcl_2 = tf.matmul(fcl_1, fcl_2_w) + fcl_2_b

y = fcl_2


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


with tf.Session() as sess:

    try:
        saver.restore(sess, "./train/dogs_cats_cnn")
        print()
        print('Initialization loaded')
        print()
    except:
        sess.run(init)
        print()
        print('New initialization done')
        print()

    for epoch in range(201):

        train_data, train_label = shuffle(train_data, train_label)

        accum_loss = 0
        accum_acc = 0

        for m in range(0, train_data.shape[0], batch):
            m2 = min(train_data.shape[0], m+batch)
            
            _, loss_scr, acc_scr = sess.run((train_step, loss, accuracy), \
                feed_dict = {x: train_data[m:m2], y_gt: train_label[m:m2], \
                keep_prob: dropout_prob})

            accum_loss += loss_scr*(m2-m)
            accum_acc += acc_scr*(m2-m)

        if epoch%10 == 0:
            print("Epoch: {}".format(epoch))
            print("Train loss = {}".format(accum_loss/train_data.shape[0]))
            print("Train accuracy = {:03.4f}".format(accum_acc/train_data.shape[0]))

            accum_acc = 0

            for m in range(0, val_data.shape[0], batch):
                m2 = min(val_data.shape[0], m+batch)
                
                acc_scr = sess.run((accuracy), \
                    feed_dict = {x: val_data[m:m2], y_gt: val_label[m:m2], \
                    keep_prob: 1})

                accum_acc += acc_scr*(m2-m)
            print("Validation accuracy = {:03.4f}".format(accum_acc/val_data.shape[0]))
            print()

        save_path = saver.save(sess, "./train/dogs_cats_cnn")


#%%

print("This is the end of the training")
print("Entering in testing mode")
print()

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
