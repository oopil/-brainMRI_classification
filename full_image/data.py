import sys
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
sys.path.append('..')
sys.path.append('/home/soopil/Desktop/Dataset/github/brainMRI_classification/ConvNeuralNet')
from data_merge import *

def _read_py_function_hippo_patch(path, label, is_masking=False, patch_size = 48, is_decode=True, is_aug = True):
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
        return (max_pos + min_pos) // 2

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

def dataloader():
    # -------- Read data ---------#
    train_x, train_t = mnist_reader.load_mnist('data/', kind='train')
    test_x, test_t = mnist_reader.load_mnist('data/', kind='t10k')
    # ------ Preprocess data -----#
    # x_train = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
    x_train = train_x
    x_train_norm = x_train / 255.0
    # x_test = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')
    x_test = test_x
    x_test_norm = x_test / 255.0
    print(np.shape(train_x), np.shape(train_t))
    print(np.shape(x_train_norm), np.shape(train_t))
    return x_train_norm, train_t, x_test_norm, test_t

def define_dataset(tr_x, tr_y, batch_size, buffer_size):
    # dataset1 = tf.data.Dataset.from_tensor_slices(tr_x)
    # dataset2 = tf.data.Dataset.from_tensor_slices(tr_y)
    # dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    # print(dataset1.output_types)  # ==> "tf.float32"
    # print(dataset1.output_shapes)  # ==> "(10,)"
    #
    # print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
    # print(dataset2.output_shapes)  # ==> "((), (100,))"
    #
    # print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    # print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

    dataset = tf.data.Dataset.from_tensor_slices(
        {"x": tr_x,
         "y": tr_y})
    print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
    print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element, iterator


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
    class_num = 2
    fold_num = 5
    sampling_option = "SIMPLE"
    whole_set = CNN_dataloader(base_folder_path, diag_type, class_option, excel_path, fold_num)
    return whole_set
