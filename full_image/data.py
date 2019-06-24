import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import binary_erosion

#######################################################################
### for reading patched mri data
#######################################################################
def mask_dilation(image_patch, label_patch, label_num, patch_size):
    dilation_iter = 3
    empty_space_shape = [patch_size for i in range(3)]
    label_pos = np.where(label_patch == label_num)
    mask_label = np.zeros(empty_space_shape)
    mask_label[label_pos] = label_num
    mask_label = binary_dilation(mask_label, iterations=dilation_iter).astype(mask_label.dtype)
    image_patch[np.where(mask_label == 0)] = 0
    return image_patch

def _read_py_function_hippo_patch(path, label, is_masking=False, patch_size = 48, is_decode=True, is_aug = True):
    isp = False
    if isp:print("file path : {}" .format(path))
    if is_decode:
        path_decoded = path.decode()
    else:
        path_decoded = path
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)
    hs = patch_size // 2

    left_subcort = [4, 5, 7, 10, 11, 12, 13, 17, 18, 26]  # 14 -(6,25,30,28)
    right_subcort = [43, 44, 46, 49, 50, 51, 52, 53, 54, 58]  # 14 -(45,57,62,60)

    index = 0
    lh_hippo = left_subcort[index] # 17
    rh_hippo = right_subcort[index] # 53
    label_list = [lh_hippo, rh_hippo]
    patch_list = []
    for label_num in label_list:
        x,y,z = label_size_check(label_array, label_num, isp)
        # transition augmentation
        if is_aug:
            ran = np.random.randint(6, size=(3))
            x += ran[0]
            y += ran[1]
            z += ran[2]

        image_patch = array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
        if is_masking:
            label_patch = label_array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
            image_patch = mask_dilation(image_patch, label_patch, label_num, patch_size)
        patch_list.append(image_patch)
    patch_array = np.concatenate(patch_list, axis=0)
    if isp: print(patch_array.shape, type(patch_array))
    patch_array = np.expand_dims(patch_array, 3)
    if is_decode:
        return patch_array.astype(np.float32), label.astype(np.int32)
    else:
        return patch_array.astype(np.float32), label

def _read_py_function_subcort_patch(path, label, is_masking=False, patch_size = 16):
    # 46 is the label of cerebellum
    left_subcort = [4, 5, 7, 10, 11, 12, 13, 17, 18, 26]  # 14 -(6,25,30,28)
    right_subcort = [43, 44, 46, 49, 50, 51, 52, 53, 54, 58]  # 14 -(45,57,62,60)
    subcort = left_subcort + right_subcort

    isp = False
    if isp:print("file path : {}" .format(path))
    path_decoded = path.decode()
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)
    hs = patch_size // 2
    label_list = subcort # 28 labels
    patch_list = []
    for label_num in label_list:
        x,y,z = label_size_check(label_array, label_num, isp)
        image_patch = array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
        if is_masking:
            label_patch = label_array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
            image_patch = mask_dilation(image_patch, label_patch, label_num, patch_size)
        patch_list.append(image_patch)
    patch_array = np.concatenate(patch_list, axis=0)
    if isp: print(patch_array.shape, type(patch_array))
    patch_array = np.expand_dims(patch_array, 3)
    return patch_array.astype(np.float32), label.astype(np.int32)

def _read_py_function_hippo_cort_patch(path, label, is_masking=False, patch_size = 16):
    # 46 is the label of cerebellum
    left_subcort = [4, 5, 7, 10, 11, 12, 13, 17, 18, 26]  # 14 -(6,25,30,28)
    right_subcort = [43, 44, 46, 49, 50, 51, 52, 53, 54, 58]  # 14 -(45,57,62,60)
    left_cort = [1000.0, 1002.0, 1003.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1011.0,
                 1012.0, 1013.0, 1014.0, 1015.0, 1016.0, 1017.0, 1018.0, 1019.0, 1020.0, 1021.0,
                 1022.0, 1023.0, 1024.0, 1025.0, 1026.0, 1027.0, 1028.0, 1029.0, 1030.0, 1031.0, 1034.0, 1035.0]  # 32
    right_cort = [2000.0, 2002.0, 2003.0, 2005.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2011.0,
                  2012.0, 2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0,
                  2022.0, 2023.0, 2024.0, 2025.0, 2026.0, 2027.0, 2028.0, 2029.0, 2030.0, 2031.0, 2034.0, 2035.0] # 32
    cort = left_cort + right_cort # too big ...
    # cort = [e for i, e in enumerate(cort) if i % 2 == 0] # half
    # subcort = left_subcort + right_subcort

    isp = False
    if isp:print("file path : {}" .format(path))
    path_decoded = path.decode()
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)
    hs = patch_size // 2
    label_list = [17,53] + cort # 35 + 35 + 2 labels
    patch_list = []
    for label_num in label_list:
        x,y,z = label_size_check(label_array, label_num, isp)
        image_patch = array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
        if is_masking:
            label_patch = label_array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
            image_patch = mask_dilation(image_patch, label_patch, label_num, patch_size)
        patch_list.append(image_patch)
    patch_array = np.concatenate(patch_list, axis=0)
    if isp: print(patch_array.shape, type(patch_array))
    patch_array = np.expand_dims(patch_array, 3)
    return patch_array.astype(np.float32), label.astype(np.int32)


def _read_py_function_1_patch(path, label, is_masking=False):
    '''
    left part : 6 ~ 30 / 1000 ~
    right part : 43 ~ 62 / 2000 ~

    use only when we need to extract some patches.
    [1000.0, 1002.0, 1003.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1011.0,
    1012.0, 1013.0, 1014.0, 1015.0, 1016.0, 1017.0, 1018.0, 1019.0, 1020.0, 1021.0,
    1022.0, 1023.0, 1024.0, 1025.0, 1026.0, 1027.0, 1028.0, 1029.0, 1030.0, 1031.0, 1034.0, 1035.0]

    [2000.0, 2002.0, 2003.0, 2005.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2011.0,
    2012.0, 2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0,
    2022.0, 2023.0, 2024.0, 2025.0, 2026.0, 2027.0, 2028.0, 2029.0, 2030.0, 2031.0, 2034.0, 2035.0]

    [2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    24.0, 26.0, 28.0, 30.0, 31.0, 41.0, 42.0, 43.0, 44.0, 46.0, 47.0, 49.0, 50.0,
    51.0, 52.0, 53.0, 54.0, 58.0, 60.0, 62.0, 63.0, 77.0, 85.0, 251.0, 252.0, 253.0, 254.0, 255.0]
    '''

    # 46 is the label of cerebellum
    left_subcort = [4, 5, 7, 10, 11, 12, 13, 17, 18, 26]  # 14 -(6,25,30,28)
    right_subcort = [43, 44, 46, 49, 50, 51, 52, 53, 54, 58]  # 14 -(45,57,62,60)
    left_cort = [1000.0, 1002.0, 1003.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1011.0,
                 1012.0, 1013.0, 1014.0, 1015.0, 1016.0, 1017.0, 1018.0, 1019.0, 1020.0, 1021.0,
                 1022.0, 1023.0, 1024.0, 1025.0, 1026.0, 1027.0, 1028.0, 1029.0, 1030.0, 1031.0, 1034.0, 1035.0]  # 32
    right_cort = [2000.0, 2002.0, 2003.0, 2005.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2011.0,
                  2012.0, 2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0,
                  2022.0, 2023.0, 2024.0, 2025.0, 2026.0, 2027.0, 2028.0, 2029.0, 2030.0, 2031.0, 2034.0, 2035.0] # 32
    cort = left_cort + right_cort
    subcort = left_subcort + right_subcort

    isp = False
    if isp:print("file path : {}" .format(path))
    path_decoded = path.decode()
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)

    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)

    # find the patch position
    patch_size = 16
    hs = patch_size // 2

    # lh_hippo = 17
    # rh_hippo = 53
    # label_list = [lh_hippo, rh_hippo]
    label_list = subcort # 28 labels
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
    # print(position_array[0], label_num)
    # print(position_array)
    # print(np.amax(position_array, axis=1))
    max_pos = np.amax(position_array, axis=1)
    min_pos = np.amin(position_array, axis=1)
    if isp: print('label square size  {}'.format(max_pos - min_pos))
    return (max_pos+min_pos)//2

def get_patch_dataset(img_l, label_l, buffer_scale = 3, is_masking=False, batch_size = 1):
    patch_read_func = _read_py_function_hippo_patch
    # patch_read_func = _read_py_function_subcort_patch
    # patch_read_func = _read_py_function_hippo_cort_patch
    # patch_read_func = _read_py_function_1_patch

    print(type(img_l), np.shape(img_l))
    mask_l = [False for _ in range(len(label_l))]
    if is_masking:
        mask_l = [True for _ in range(len(label_l))]
    dataset = tf.data.Dataset.from_tensor_slices((img_l, label_l, mask_l))
    dataset = dataset.map(lambda img_l, label_l, mask_l:
        tuple(tf.py_func(patch_read_func, [img_l, label_l, mask_l], [tf.float32, tf.int32])), num_parallel_calls=5)
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

def get_patch_dataset_handler(img_l, label_l, buffer_scale = 3, is_masking=False, batch_size = 1):
    print(type(img_l), np.shape(img_l))
    mask_l = [False for _ in range(len(label_l))]
    if is_masking:
        mask_l = [True for _ in range(len(label_l))]
    dataset = tf.data.Dataset.from_tensor_slices((img_l, label_l, mask_l))
    dataset = dataset.map(lambda img_l, label_l, mask_l:
                          tuple(tf.py_func(_read_py_function_1_patch, [img_l, label_l, mask_l], [tf.float32, tf.int32])), num_parallel_calls=5)
    dataset = dataset.shuffle(buffer_size=(int(len(img_l)* 0.4) + buffer_scale * batch_size)).repeat().batch(batch_size)
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

def read_test_data(img_l, label_l, is_masking=False):
    patch_read_func = _read_py_function_hippo_patch
    # patch_read_func = _read_py_function_subcort_patch
    # patch_read_func = _read_py_function_hippo_cort_patch
    # patch_read_func = _read_py_function_1_patch

    assert len(img_l) == len(label_l)
    length = len(label_l)

    test_data, test_label = [], []
    for i in range(length):
        d, c = patch_read_func(img_l[i], label_l[i], is_masking=False, patch_size = 48, is_decode=False, is_aug=False)
        test_data.append(d)
        test_label.append(c)
        # print(i, img_l[i], label_l[i])
    # print(length, np.shape(test_data), np.shape(test_label))
    # assert False
    return np.array(test_data), np.array(test_label)
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