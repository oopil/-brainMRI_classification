import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation

#######################################################################
### for reading patched mri data
#######################################################################
"""
def _read_py_function_1_patch(path, label):
    '''
    use only when we need to extract some patches.
    '''
    isp = False
    if isp:print("file path : {}" .format(path))
    path_decoded = path.decode()
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)

    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)

    # find the patch position
    patch_size = 48
    hs = patch_size // 2
    x1,y1,z1 = label_size_check(label_array, 17, isp)
    x2,y2,z2 = label_size_check(label_array, 53, isp)
    # extract patch and concatenate
    patch_array = np.concatenate((array[x1-hs:x1+hs,y1-hs:y1+hs,z1-hs:z1+hs],\
                                  array[x2-hs:x2+hs,y2-hs:y2+hs,z2-hs:z2+hs]),axis=0)
    #normalize
    if isp: print(patch_array.shape, type(patch_array))
    patch_array = np.expand_dims(patch_array, 3)
    return patch_array.astype(np.float32), label.astype(np.int32)
"""
def mask_dilation(image_patch, label_patch, label_num, patch_size):
    dilation_iter = 2
    empty_space_shape = [patch_size for i in range(3)]
    label_pos = np.where(label_patch == label_num)
    mask_label = np.zeros(empty_space_shape)
    mask_label[label_pos] = label_num
    mask_label = binary_dilation(mask_label, iterations=dilation_iter).astype(
        mask_label.dtype)
    image_patch[np.where(mask_label == 0)] = 0
    return image_patch

def _read_py_function_1_patch(path, label, is_masking=False):
    '''
    use only when we need to extract some patches.
    '''
    isp = False
    if isp:print("file path : {}" .format(path))
    path_decoded = path.decode()
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)

    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)

    # find the patch position
    patch_size = 48
    hs = patch_size // 2

    lh_hippo = 17
    rh_hippo = 53
    label_list = [lh_hippo, rh_hippo]
    patch_list = []
    for label_num in label_list:
        x,y,z = label_size_check(label_array, label_num, isp)
        image_patch = array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
        if is_masking:
            label_patch = label_array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
            image_patch = mask_dilation(image_patch, label_patch, label_num, patch_size)
        patch_list.append(image_patch)

    patch_array = np.concatenate(patch_list, axis=0)
    #normalize
    if isp: print(patch_array.shape, type(patch_array))
    patch_array = np.expand_dims(patch_array, 3)
    return patch_array.astype(np.float32), label.astype(np.int32)

def label_size_check(label_array, label_num, isp):
    '''
    print the size of label square
    :return: return the center position
    '''
    # print(label_array, label_num)
    position_array = np.where(label_array == label_num)
    # print(position_array)
    # print(np.amax(position_array, axis=1))
    max_pos = np.amax(position_array, axis=1)
    min_pos = np.amin(position_array, axis=1)
    if isp: print('label square size  {}'.format(max_pos - min_pos))
    return (max_pos+min_pos)//2

def get_patch_dataset(img_l, label_l, buffer_scale = 3, is_masking=False, batch_size = 1):
    print(type(img_l), np.shape(img_l))
    mask_l = [False for _ in range(len(label_l))]
    if is_masking:
        mask_l = [True for _ in range(len(label_l))]
    dataset = tf.data.Dataset.from_tensor_slices((img_l, label_l, mask_l))
    dataset = dataset.map(lambda img_l, label_l, mask_l: tuple(tf.py_func(_read_py_function_1_patch, [img_l, label_l, mask_l], [tf.float32, tf.int32])), num_parallel_calls=8)
    # dataset = dataset.shuffle(buffer_size=(int(len(img_l)* 0.4) + buffer_scale * batch_size)).batch(batch_size)
    dataset = dataset.shuffle(buffer_size=(int(len(img_l)* 0.4) + buffer_scale * batch_size)).repeat().batch(batch_size)
    # dataset = dataset.shuffle(buffer_size=(int(len(img_l)* 0.4) + 3 * batch_size))
    # dataset = dataset.shuffle(buffer_size=(len(img_l) * batch_size))
    # dataset = dataset.shuffle(buffer_size=batch_size)
    print(dataset)

    # handle = tf.placeholder(tf.string, shape=[])
    iterator = dataset.make_initializable_iterator()
    # iterator = tf.data.Iterator.from_string_handle(
    #     handle, dataset.output_types, ([None, 212, 320, 240, 1], [None, 1]))  # image dimension[212, 320, 240]
    next_element = iterator.get_next()
    return next_element, iterator

def normalize_np(X_):
    print('normalize the data ... ')
    print(np.amax(X_), np.amin(X_))
    return (X_-np.amin(X_))/np.amax(X_)

# def read_patch_no
#######################################################################
### for reading whole mri data
#######################################################################
def _read_py_function(img_path, label):
    img_path_decoded = img_path.decode() # what the fuck !!! careful about decoding
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    # print(array.shape, type(array))
    # img = tf.img.decode_jpeg(img_path)
    # label = tf.img.decode_png(label_path)
    array = np.expand_dims(array, 3)
    return array.astype(np.float32), label.astype(np.int32)

def get_dataset(img_l, label_l, batch_size=1):
    # dataset = tf.data.Dataset.from_tensor_slices((img_l, label_l))
    dataset = tf.data.Dataset.from_tensor_slices((img_l, label_l))
    dataset = dataset.map(lambda img_l, label_l: tuple(tf.py_func(_read_py_function, [img_l, label_l], [tf.float32, tf.int32])))
    # dataset = dataset.repeat()
    # dataset = dataset.shuffle(buffer_size=(int(len(img_l)* 0.4) + 3 * batch_size))
    dataset = dataset.shuffle(buffer_size=(len(img_l) * batch_size))
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

#######################################################################
### Rest of them
#######################################################################
def column_to_list(matrix, i, num):
    return [row[i] for row in matrix]

def test():
    path = '/home/sp/Datasets/MRI_chosun/test_sample_2/aAD/T1sag/14092806/T1.nii.gz'
    array, label = _read_py_function(path, 0)
    print(array)

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

def check_image_shape(path):
    itk_file = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(itk_file)
    array_shape = array.shape
    return array_shape

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