import tensorflow as tf
import os
from random import randint
import numpy as np
import SimpleITK as sitk
import os
from time import time
import pickle
from random import shuffle

from brats_3d_seg import build_model

FLAGS = tf.app.flags.FLAGS


def get_files():
    hgg = "HGG"
    lgg = "LGG"
    path = FLAGS.data_dir+"/Brats17TrainingData"
    try:
        data = load_files_checkpoint(FLAGS.files_checkpoint)
    except:
        hgg_sub = os.listdir(os.path.join(path,hgg))
        hgg_sub = [hgg+","+sub for sub in hgg_sub]
        lgg_sub = os.listdir(os.path.join(path,lgg))
        lgg_sub = [lgg+","+sub for sub in lgg_sub]
        data = {}
        data_list = hgg_sub + lgg_sub
        shuffle(data_list)
        n = int(len(data_list)*0.9)
        data["train"] = data_list[:n]
        data["val"] = data_list[n:]
        save_files_checkpoint(data, FLAGS.files_checkpoint)
    data["train"] = [path+","+sub for sub in data["train"]]
    return data["train"]


def load_files_checkpoint(path):
    files = {}
    with open(path, 'rb') as handle:
        files = pickle.load(handle)
        handle.close()
    print('File ' + path + ' loaded\n')
    return files

def save_files_checkpoint(files, path):
    with open(path, 'wb') as handle:
        pickle.dump(files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    print('File ' + path + ' saved\n')


def load_data(filename):
    pz = 8
    py = 24
    px = 24
    my_file = filename.decode().split(",")
    path = my_file[0]
    folder = my_file[1]
    subject = my_file[2]
    ext_list = ["_seg.nii.gz","_flair.nii.gz","_t1.nii.gz",
        "_t1ce.nii.gz","_t2.nii.gz"]
    sub_arrays = []
    arrays_dict = {}
    for n in range(len(ext_list)):
        file_name = os.path.join(path,folder,subject,subject+ext_list[n])
        img = sitk.ReadImage(file_name)
        sub_arrays.append(sitk.GetArrayFromImage(img))
        arrays_dict[n] = []
    m = 0
    while m < 32:
        x = randint(0,240-px)
        y = randint(0,240-py)
        z = randint(0,155-pz)
        if np.sum(sub_arrays[0][z:z+pz,y:y+py,x:x+px])>=0:
            m += 1
            for n in range(len(ext_list)):
                arrays_dict[n].append(sub_arrays[n][z:z+pz,y:y+py,x:x+px])
    for n in range(len(ext_list)):
        arrays_dict[n] = np.array(arrays_dict[n])
    label = arrays_dict[0]
    label[label>3] = 3
    array = np.concatenate((np.expand_dims(arrays_dict[1], axis=3),
        np.expand_dims(arrays_dict[2], axis=3),
        np.expand_dims(arrays_dict[3], axis=3),
        np.expand_dims(arrays_dict[4], axis=3)),
        axis=3)
    return array.astype(np.float32), label.astype(np.int32)


def get_dataset(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(
        lambda filename: tuple(tf.py_func(
                load_data, [filename], [tf.float32,tf.int32])),
                num_parallel_calls=5)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.batch(FLAGS.batch_size)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
            handle, dataset.output_types, ([None,32,8,24,24,4],[None,32,8,24,24]))
    next_element = iterator.get_next()
    iterator = dataset.make_one_shot_iterator()
    return next_element, iterator, handle


def train():
    files = get_files()
    next_element, iterator, handle = get_dataset(files)

    x = tf.reshape(next_element[0], shape=[128,8,24,24,4])
    y_gt = tf.reshape(next_element[1], shape=[128,8,24,24])
    y_gt = tf.one_hot(y_gt, 4)

    net = build_model(inputs=x, labels=y_gt)

    rate = tf.placeholder(dtype=tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    train = optimizer.minimize(net["loss"])

    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)
    data_handle = sess.run(iterator.string_handle())
    new_rate = FLAGS.learning_rate
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint)

    try:
        checkpoints = FLAGS.test_checkpoints.split(",")
        saver.restore(sess, FLAGS.checkpoint_path+'-'+checkpoints[0])
        start = int(checkpoints[0])
        end = start + FLAGS.max_steps
        rate_updates = int(start/FLAGS.steps_to_learning_rate_update)
        for n in range(rate_updates):
            new_rate = new_rate * (1-FLAGS.learning_rate_decrease)
        print("\nCheckpoint {} loaded".format(checkpoints[0]))
        print("New learning rate = {}\n".format(new_rate))
    except:
        start = 0
        end = FLAGS.max_steps
        print("\nNew Initialization\n")
        pass

    print()
    for n in range(start, end):
        t1 = time()
        _, s_loss, s_dsc = sess.run((train,
            net["loss"],
            net["dsc"]),
            feed_dict={handle: data_handle, rate: new_rate})
        t2 = time()
        train_status = 'loss:{:0.4f} - tc:{:0.3f} wt:{:0.3f} et:{:0.3f} '+ \
            'bgd:{:0.3f} - time:{:0.3f} - step:{}/{}'
        print(train_status.format(s_loss, s_dsc[1], s_dsc[2], s_dsc[3],
            s_dsc[0], t2-t1, n+1, FLAGS.max_steps))

        if (n+1)%FLAGS.steps_to_learning_rate_update == 0:
            new_rate = new_rate * (1-FLAGS.learning_rate_decrease)
            print('New learning rate = {}'.format(new_rate))
            print()

        if (n+1)%FLAGS.steps_to_save_checkpoint == 0:
            saver.save(sess,
            	FLAGS.checkpoint_path,
                global_step=n+1,
                write_meta_graph=False)
            print('Check point saved')
            print()


def main():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.cuda_device
    train()
    print("done!")


if __name__ == '__main__':
    main()
