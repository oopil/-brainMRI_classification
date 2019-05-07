import os
import sys
sys.path.append('/home/soopil/Desktop/github/brainMRI_classification')
sys.path.append('/home/soopil/Desktop/github/brainMRI_classification/sample_image')
import numpy as np
import SimpleITK as sitk
# /home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed

def _read_py_function_1_patch(path, label):
    '''
    use only when we need to extract some patches.
    '''
    isp = False
    if isp:print("file path : {}" .format(path))
    path_decoded = path.decode()
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    # img_path_decoded = img_path.decode()
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)

    # label_path_decoded = label_path.decode()
    label_itk_file = sitk.ReadImage(label_path_decoded)
    label_array = sitk.GetArrayFromImage(label_itk_file)

    # find the patch position
    patch_size = 48
    hs = patch_size // 2
    x1,y1,z1 = label_size_check(label_array, 17, isp)
    x2,y2,z2 = label_size_check(label_array, 53, isp)
    # p1= array[x1-hs:x1+hs,y1-hs:y1+hs,z1-hs:z1+hs]
    # p2= array[x2-hs:x2+hs,y2-hs:y2+hs,z2-hs:z2+hs]
    patch_array = np.concatenate((array[x1-hs:x1+hs,y1-hs:y1+hs,z1-hs:z1+hs],\
                                  array[x2-hs:x2+hs,y2-hs:y2+hs,z2-hs:z2+hs]),axis=0)
    # extract patch and concatenate
    if isp: print(patch_array.shape, type(patch_array))
    array = np.expand_dims(array, 3)
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

def read_MRI(img_path, label):
    print("file path : {}" .format(img_path))
    img_path_decoded = img_path #.decode() # what the fuck !!! careful about decoding
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    print(array.shape, type(array))
    # array = np.expand_dims(array, 3)
    return array.astype(np.float32), itk_file

def count_label_num(label_array):
    label_list = []
    # print(np.where(label_array))
    # print(len(np.where(label_array)[0]))
    position_array = np.where(label_array)
    for i in range(len(position_array[0])):
        label_list.append(label_array[position_array[0][i], position_array[1][i], position_array[2][i]])
    label_list = list(set(label_list))
    print('Total label number : {}'.format(len(label_list)))
    return label_list

def draw_patch_box(space, center, size, label):
    x,y,z = center
    hs = size//2
    space[x - hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x + hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z - hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z + hs] = label
    space[x - hs:x + hs, y - hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y + hs, z - hs:z + hs] = label

    hs -= 1
    space[x - hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x + hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z - hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z + hs] = label
    space[x - hs:x + hs, y - hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y + hs, z - hs:z + hs] = label

    hs += 1
    space[x - hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x + hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z - hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z + hs] = label
    space[x - hs:x + hs, y - hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y + hs, z - hs:z + hs] = label
    return space

def check_mri_label():
    print('start MRI label check.')
    sample_dir_path = '/home/sp/PycharmProjects/brainMRI_classification/sample_image/ADDlabel'
    file_name_str = 'T1Label.nii.gz  aparc.DKTatlas+aseg.nii  aseg.auto.nii aparc+aseg.nii  aparc.a2009s+aseg.nii'
    file_name = [ e for e in file_name_str.split(' ') if e != '']
    print('label file : ',file_name)
    isp = True
    orig_file = file_name[0]
    label = 0 # ADD
    orig_file_path = os.path.join(sample_dir_path, orig_file)
    orig_array, itk_file = read_MRI(orig_file_path, label)
    orig_label_list = count_label_num(orig_array)
    '''
    for i in range(1,5):
    new_file = file_name[i]
    new_file_path = os.path.join(sample_dir_path, new_file)
    # new_file_path = file
    new_array, _ = read_MRI(new_file_path, label)
    new_label_list = count_label_num(new_array)
    assert False
    '''

    '''
    in this case, we use
    aparc.DKTatlas+aseg.nii
    17,53 : left and right hippocampus label
    '''
    new_file = file_name[1]
    new_file_path = os.path.join(sample_dir_path, new_file)
    # new_file_path = file
    new_array, itk_file = read_MRI(new_file_path, label)
    new_label_list = count_label_num(new_array)

    # for i in range(len(new_label_list)):
    #     if not new_label_list[i] in orig_label_list:
    #         print(new_label_list[i])
    empty_space_shape = [256 for i in range(3)]
    empty_space = np.zeros(empty_space_shape)
    # draw_array = new_array
    draw_array = empty_space
    label_color = 1 # red
    patch_size = 48
    center_pos = label_size_check(new_array, 17, isp)
    draw_array = draw_patch_box(draw_array, center_pos, patch_size, label=label_color)
    center_pos = label_size_check(new_array, 53, isp)
    draw_array = draw_patch_box(draw_array, center_pos, patch_size, label=label_color)
    # label_size_check(new_array, 18)
    # label_size_check(new_array, 54)

    draw_file_path = os.path.join(sample_dir_path, 'draw_patch_only.nii')
    draw_file = sitk.GetImageFromArray(draw_array)
    draw_file.CopyInformation(itk_file)
    sitk.WriteImage(draw_file, draw_file_path)
    print('saved the file : {}'.format(draw_file_path))

def main():
    check_mri_label()

if __name__ == '__main__':
    main()

