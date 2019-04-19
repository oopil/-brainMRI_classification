import tensorflow as tf
import SimpleITK as sitk
import numpy as np
from class_metabot import *
from chosun_pipeline import *
from data import *

def _read_py_function(img_path, label):
    img_path_decoded = img_path.decode() # what the fuck !!! careful about decoding
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    # print(array.shape, type(array))
    # img = tf.img.decode_jpeg(img_path)
    # label = tf.img.decode_png(label_path)

    array = np.expand_dims(array, 3)
    return array.astype(np.float32), label.astype(np.int32)
    # return array.astype(np.float32)

def pathloader(index : int)->list:
    loader = DataLoader(0)
    class_name = ['aAD', 'ADD', 'mAD', 'NC']
    # we need to set the dataset path correctly. in the chosun_pipeline.py file.
    meta_data_list = extr_meta_data(class_name[index])

    total_num = len(meta_data_list)
    for line in meta_data_list:
        print(line)

    path_list = []
    # sample : ['/home/sp/Datasets/MRI_chosun/test_sample_2/aAD/T1sag', '14062105', '/home/sp/Datasets/MRI_chosun/test_sample_2/aAD/T1sag/14062105/T1.nii.gz']
    for i, subj in enumerate(meta_data_list):
        subj_name = subj[1]
        folder_path = subj[0]
        subj_dir_path = os.path.join(folder_path, subj_name)
        input_file = 'T1.nii.gz'
        # input_file = 'freesurfer/mri/nifti/brain.nii'
        # label_file = 'freesurfer/mri/nifti/auto_aseg.nii'
        input_path = os.path.join(subj_dir_path,input_file)
        path_list.append(input_path)
        # print('<<< Index : ', i+1, '/', total_num ,' >>>')
        # chosun_MRI_preprocess(subj[0], subj[1], subj[2])
    return path_list

def test():
    path = '/home/sp/Datasets/MRI_chosun/test_sample_2/aAD/T1sag/14092806/T1.nii.gz'
    array, label = _read_py_function(path, 0)
    print(array)

def get_dataset(img_l, label_l):
    batch_size = 2
    # dataset = tf.data.Dataset.from_tensor_slices((img_l, label_l))
    dataset = tf.data.Dataset.from_tensor_slices((img_l, label_l))
    dataset = dataset.map(lambda img_l, label_l: tuple(tf.py_func(_read_py_function, [img_l, label_l], [tf.float32, tf.int32])))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=(int(len(img_l)* 0.4) + 3 * batch_size))
    dataset = dataset.batch(batch_size)
    print(dataset)

    # handle = tf.placeholder(tf.string, shape=[])
    iterator = dataset.make_initializable_iterator()
    # iterator = tf.data.Iterator.from_string_handle(
    #     handle, dataset.output_types, ([None, 212, 320, 240, 1], [None, 1]))  # image dimension[212, 320, 240]
    next_element = iterator.get_next()
    return next_element, iterator

    # handle = tf.placeholder(tf.string, shape=[])
    # # iterator = dataset.make_initializable_iterator()
    # iterator = tf.data.Iterator.from_string_handle(
    #     handle, dataset.output_types, ([None, 212, 320, 240, 1], [None, 1])) # image dimension[212, 320, 240]
    # img_stacked, label_stacked = iterator.get_next()
    # next_element = iterator.get_next()
    # iterator = dataset.make_one_shot_iterator()
    # # print(img_stacked, label_stacked)
    # return next_element, iterator, handle

if __name__ == '__main__':
    # test()
    # assert False

    is_merge = True  # True
    option_num = 0  # P V T options
    class_option = ['NC vs AD','NC vs MCI','MCI vs AD','NC vs MCI vs AD']
    class_option_index = 0
    class_num = class_option_index//3 + 2

    ford_num = 10
    ford_index = 0
    keep_prob = 0.9 # 0.7

    learning_rate = 0.02
    epochs = 2000
    print_freq = 100
    save_freq = 300

    # data, label = dataloader(class_option[class_option_index], option_num, is_merge=is_merge)
    # # assert False
    # data, label = shuffle_two_arrays(data, label)
    # X_train, Y_train, X_test, Y_test = split_train_test(data, label, ford_num, ford_index)
    # # print(len(data[0]), len(X_train[0]))
    # X_train, Y_train = valence_class(X_train, Y_train, class_num)
    # X_test, Y_test = valence_class(X_test, Y_test, class_num)
    # feature_num = len(X_train[0])
    # print(X_train.shape, X_test.shape)

    img_l = pathloader(0)
    label_l = []
    tmp_label_l = [0,1,0]

    next_element, iterator = get_dataset(img_l, tmp_label_l)
    # next_element, iterator, handle = get_dataset(img_l, tmp_label_l)
    print(next_element[0].shape)
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        my_array, my_label = sess.run(next_element)
        print(my_array.shape, my_label.shape)

        # data_handle = sess.run(iterator.string_handle())
        # print(data_handle)
        # my_array = sess.run(next_element[0], feed_dict={handle: data_handle})
        # print(my_array)


def load_nii_data(path):
    '''
    essential apis to use nifti file.
    :param path:
    :return:
    '''
    result_path = 'eq_' + path
    itk_file = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(itk_file)
    # array_eq = exposure.equalize_hist(array)
    # min_intensity = array_eq.min(axis=(0,1,2), keepdims=True)
    # max_intensity = array_eq.max(axis=(0, 1, 2), keepdims=True)
    #
    # array_eq_normal = array_eq *(max_intensity/(max_intensity-min_intensity))
    # print(min_intensity, max_intensity)
    shape_arr = array.shape

    slice1 = int(shape_arr[0]/2)
    slice2 = int(shape_arr[1]/ 2)
    slice3 = int(shape_arr[2]/ 2)
    print(array[slice1:slice1+1,slice2:slice2+1, slice3:slice3+5])
    # print(array_eq[slice1:slice1 + 1, slice2:slice2 + 1, slice3:slice3+5])
    # print(array_eq_normal[slice1:slice1+1,slice2:slice2+1, slice3:slice3+5])
    # print()
    # new_file = sitk.GetImageFromArray(array_eq)
    # new_file.CopyInformation(itk_file)
    # sitk.WriteImage(new_file, result_path)