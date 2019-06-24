import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from data_merge import *
import argparse

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
    parser.add_argument('--setting',            default='desktop', type=str) # desktop sv186 sv202 sv144
    return parser.parse_args()

def setting(sv_set):
    base_folder_path = ''
    excel_path = ''
    base_path = ''
    if sv_set == 186:
        base_folder_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_empty_copy'
        excel_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 0:  # desktop
        base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
        base_path = '/home/soopil/Desktop/Dataset/MRI_chosun'
    elif sv_set == 202:
        base_folder_path = '/home/soopil/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        excel_path = '/home/soopil/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
        base_path = '/home/soopil/Datasets/MRI_chosun'
    else:
        assert False
    return base_folder_path, excel_path, base_path

def _read_py_function_hippo_patch(path, label, is_masking=False):
    def label_size_check(label_array, label_num, isp):
        '''
        print the size of label square
        :return: return the center position
        '''
        position_array = np.where(label_array == label_num)
        max_pos = np.amax(position_array, axis=1)
        min_pos = np.amin(position_array, axis=1)
        if isp: print('label square size  {}'.format(max_pos - min_pos))
        return (max_pos + min_pos) // 2

    '''
    use only when we need to extract some patches.
    '''
    isp = False
    if isp:print("file path : {}" .format(path))
    # path_decoded = path.decode()
    img_path_decoded, label_path_decoded = path.split(',')
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
            # image_patch = mask_dilation(image_patch, label_patch, label_num, patch_size)
        patch_list.append(image_patch)

    patch_array = np.concatenate(patch_list, axis=0)
    #normalize
    if isp: print(patch_array.shape, type(patch_array))
    # patch_array = np.expand_dims(patch_array, 3)
    return patch_array.astype(np.float32), label.astype(np.int32)

def read_patch(path, label, brain_part):
    def label_size_check(label_array, label_num, isp):
        '''
        print the size of label square
        :return: return the center position
        '''
        position_array = np.where(label_array == label_num)
        max_pos = np.amax(position_array, axis=1)
        min_pos = np.amin(position_array, axis=1)
        if isp: print('label square size  {}'.format(max_pos - min_pos))
        return (max_pos + min_pos) // 2

    isp = False
    if isp:print("file path : {}" .format(path))
    img_path_decoded, label_path_decoded = path.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)

    # find the patch position
    patch_size = 32
    hs = patch_size // 2
    x, y, z = label_size_check(label_array, brain_part, isp)
    image_patch = array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
    return image_patch.astype(np.float32), label.astype(np.int32)

def try_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def hippo_bag(base_folder_path, excel_path, base_path):
    diag_type = "clinic"
    class_option = 'CN vs AD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    sampling_option = "None"
    class_num = 2
    patch_size = 48
    excel_option = 'merge'
    test_num = 10
    fold_num = 5
    is_split_by_num = False
    whole_set = CNN_dataloader(base_folder_path, diag_type, class_option, excel_path, fold_num)
    # whole_set = np.array(whole_set)
    train_data, train_label, test_data, test_label = whole_set[1]
    test_data, test_label = valence_class(test_data, test_label, class_num)
    # if sampling_option != "None":
    #     train_data, train_label = over_sampling(train_data, train_label, sampling_option)

    def instance_gen(array):
        s0, s1, s2 = array.shape
        sub_patch_size = 16
        # print('instance size is : {}'.format(sub_patch_size))
        instance_list = []
        position_list = []
        for i in range(0, s0, sub_patch_size):
            for j in range(0, s1, sub_patch_size):
                for k in range(0, s2, sub_patch_size):
                    instance_list.append(array[i:i + sub_patch_size, j:j + sub_patch_size, k:k + sub_patch_size])
                    # print(i, j, k)
                    position_list.append([i,j,k])
        print(position_list)
        assert False
        return instance_list

    def bag_gen(input_data,input_label, path, type = 'train'):
        base_path = path + '/'+ type
        for path, label in zip(input_data, input_label):
            img_path_decoded, label_path_decoded = path.split(',')
            dir_split = img_path_decoded.split('/')
            # print(dir_split)
            subj_name = dir_split[-3]
            dir_name = 'None'
            if label == 0:
                dir_name = 'NC'
            elif label == 1:
                dir_name = 'AD'
            file_path = os.path.join(base_path, dir_name, subj_name)

            print(subj_name, label)
            print(file_path)

            array, label = _read_py_function_hippo_patch(path, label)
            instance_list = instance_gen(array)
            print(np.shape(array))
            np.save(file_path, instance_list)
        pass

    base_path = os.path.join(base_path, 'MIL_bag/instance_size_16')
    try_mkdir(base_path)

    test_dir = os.path.join(base_path, 'test')
    train_dir = os.path.join(base_path, 'train')
    try_mkdir(test_dir)
    try_mkdir(train_dir)
    try_mkdir(os.path.join(train_dir, 'AD'))
    try_mkdir(os.path.join(train_dir, 'NC'))
    try_mkdir(os.path.join(test_dir, 'AD'))
    try_mkdir(os.path.join(test_dir, 'NC'))
    # try_mkdir(os.path.join(train_dir, 'NC'))
    # os.mkdir(test_dir)
    # os.mkdir(train_dir)
    # os.mkdir(os.path.join(train_dir, 'AD'))
    # os.mkdir(os.path.join(train_dir, 'NC'))
    # os.mkdir(os.path.join(test_dir, 'AD'))
    # os.mkdir(os.path.join(test_dir, 'NC'))
    '''
    MIL test data bag generation &&
    MIL train data bag generation
    '''
    bag_gen(train_data, train_label, path=base_path, type='train')
    bag_gen(test_data, test_label, path=base_path, type='test')

    assert False
    array, label = _read_py_function_hippo_patch(train_data[0], train_label[0])
    print(array.shape)
    s0,s1,s2 = array.shape
    sub_patch_size = 8
    instance_list = []

    for i in range(0,s0,sub_patch_size):
        for j in range(0,s1,sub_patch_size):
            for k in range(0,s2,sub_patch_size):
                instance_list.append(array[i:i+sub_patch_size,j:j+sub_patch_size,k:k+sub_patch_size])
                print(i,j,k)
    file_path = '/home/soopil/Desktop/Dataset/MRI_chosun/MIL_sample'
    np.save(file_path,instance_list)
    sample = np.load(file_path +'.npy')
    # print(sample)
    print(sample.shape)
    # patch_array = np.concatenate(instance_list, axis=0)
    # print(np.shape(instance_list))
    # print(np.shape(patch_array))
    # array = array.reshape(8,8,8,64)
    assert False
    # instances = np.array_split(array, [8,8,8])
    instances = np.array_split(array, 8, axis=2)
    print(np.shape(instances))
    instances = np.array_split(instances, 8, axis=1)
    print(np.shape(instances))
    instances = np.array_split(instances, 8, axis=0)
    print(np.shape(instances))

    # print(instances[0])
    print(np.shape(instances))
    print(np.shape(instances[0]))

def brain_bag(base_folder_path, excel_path, base_path, instance_list):
    diag_type = "clinic"
    class_option = 'CN vs AD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    sampling_option = "None"
    class_num = 2
    patch_size = 48
    excel_option = 'merge'
    test_num = 10
    fold_num = 5
    is_split_by_num = False
    whole_set = CNN_dataloader(base_folder_path, diag_type, class_option, excel_path, fold_num)
    # whole_set = np.array(whole_set)
    train_data, train_label, test_data, test_label = whole_set[1]
    test_data, test_label = valence_class(test_data, test_label, class_num)
    # if sampling_option != "None":
    #     train_data, train_label = over_sampling(train_data, train_label, sampling_option)

    print(instance_list)
    print('number of instances : ',len(instance_list))
    # assert False

    def bag_gen(input_data, input_label, path, brain_part_list ,type = 'train'):
        sub_patch_size = 32
        base_path = path + '/'+ type
        for path, label in zip(input_data, input_label):
            img_path_decoded, label_path_decoded = path.split(',')
            dir_split = img_path_decoded.split('/')
            subj_name = dir_split[-3]
            dir_name = 'None'

            if label == 0:
                dir_name = 'NC'
            elif label == 1:
                dir_name = 'AD'

            file_path = os.path.join(base_path, dir_name, subj_name)

            print(subj_name, label)
            print(file_path)

            bag = []
            for brain_part in brain_part_list:
                array, label = read_patch(path, label, brain_part)
                bag.append(array)

            bag = np.array(bag)
            print(np.shape(bag))
            # assert False
            np.save(file_path, bag)

    base_path = os.path.join(base_path, 'MIL_bag/instance_size_32')
    try_mkdir(base_path)

    test_dir = os.path.join(base_path, 'test')
    train_dir = os.path.join(base_path, 'train')

    try_mkdir(test_dir)
    try_mkdir(train_dir)
    try_mkdir(os.path.join(train_dir, 'AD'))
    try_mkdir(os.path.join(train_dir, 'NC'))
    try_mkdir(os.path.join(test_dir, 'AD'))
    try_mkdir(os.path.join(test_dir, 'NC'))
    '''
    MIL test data bag generation &&
    MIL train data bag generation
    '''
    bag_gen(train_data, train_label, path=base_path, brain_part_list=instance_list, type='train')
    bag_gen(test_data, test_label, path=base_path, brain_part_list=instance_list, type='test')

if __name__ == '__main__':
    args = parse_args()
    sv_set_dict = {
        "desktop": 0,
        "sv186": 186,
        "sv144": 144,
        "sv202": 202,
    }
    sv_set = sv_set_dict[args.setting]
    base_folder_path, excel_path, base_path = setting(sv_set)

    left_cort = [1000.0, 1002.0, 1003.0, 1005.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1011.0,
                 1012.0, 1013.0, 1014.0, 1015.0, 1016.0, 1017.0, 1018.0, 1019.0, 1020.0, 1021.0,
                 1022.0, 1023.0, 1024.0, 1025.0, 1026.0, 1027.0, 1028.0, 1029.0, 1030.0, 1031.0,
                 1034.0, 1035.0]  # 35
    right_cort = [2000.0, 2002.0, 2003.0, 2005.0, 2006.0, 2007.0, 2008.0, 2009.0, 2010.0, 2011.0,
                  2012.0, 2013.0, 2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0, 2021.0,
                  2022.0, 2023.0, 2024.0, 2025.0, 2026.0, 2027.0, 2028.0, 2029.0, 2030.0, 2031.0,
                  2034.0, 2035.0]  # 35
    left_subcort = [4, 5, 7, 10, 11, 12, 13, 17, 18, 26]  # 14 -(6,25,30,28)
    right_subcort = [43, 44, 46, 49, 50, 51, 52, 53, 54, 58]  # 14 -(45,57,62,60)
    cort = left_cort + right_cort  # too big ...
    subcort = left_subcort + right_subcort
    instance_list = subcort + cort

    # brain_bag(base_folder_path, excel_path, base_path, instance_list)
    hippo_bag(base_folder_path, excel_path, base_path)