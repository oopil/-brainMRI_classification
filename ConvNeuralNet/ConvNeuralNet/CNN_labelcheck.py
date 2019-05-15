import os
import sys
sys.path.append('/home/soopil/Desktop/github/brainMRI_classification')
sys.path.append('/home/soopil/Desktop/github/brainMRI_classification/sample_image')
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
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
    lh_hippo_pos = np.where(label_array == label_num)
    # print(lh_hippo_pos)
    max_pos = np.amax(lh_hippo_pos, axis=1)
    min_pos = np.amin(lh_hippo_pos, axis=1)
    median_pos = np.median(lh_hippo_pos, axis=1).astype(np.int32)
    if isp:
        print('label square size  {}'.format(max_pos - min_pos))
        # print(max_pos)
        # print(min_pos)
        print('median point : {}'.format(median_pos))
    # return (max_pos+min_pos)//2
    return median_pos

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
    lh_hippo_pos = np.where(label_array)
    for i in range(len(lh_hippo_pos[0])):
        label_list.append(label_array[lh_hippo_pos[0][i], lh_hippo_pos[1][i], lh_hippo_pos[2][i]])
    label_list = list(set(label_list))
    print('Total label number : {}'.format(len(label_list)))
    return label_list

def draw_patch_box(space, center, size, label, thickness = 1):
    x,y,z = center
    hs = size//2
    space[x - hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x + hs, y - hs:y + hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z - hs] = label
    space[x - hs:x + hs, y - hs:y + hs, z + hs] = label
    space[x - hs:x + hs, y - hs, z - hs:z + hs] = label
    space[x - hs:x + hs, y + hs, z - hs:z + hs] = label
    for i in range(thickness):
        hs += 1
        space[x - hs, y - hs:y + hs, z - hs:z + hs] = label
        space[x + hs, y - hs:y + hs, z - hs:z + hs] = label
        space[x - hs:x + hs, y - hs:y + hs, z - hs] = label
        space[x - hs:x + hs, y - hs:y + hs, z + hs] = label
        space[x - hs:x + hs, y - hs, z - hs:z + hs] = label
        space[x - hs:x + hs, y + hs, z - hs:z + hs] = label
    #
    # hs += 1
    # space[x - hs, y - hs:y + hs, z - hs:z + hs] = label
    # space[x + hs, y - hs:y + hs, z - hs:z + hs] = label
    # space[x - hs:x + hs, y - hs:y + hs, z - hs] = label
    # space[x - hs:x + hs, y - hs:y + hs, z + hs] = label
    # space[x - hs:x + hs, y - hs, z - hs:z + hs] = label
    # space[x - hs:x + hs, y + hs, z - hs:z + hs] = label
    return space

def save_nifti_file(draw_array, itk_file, sample_dir_path, save_file_name):
    draw_file_path = os.path.join(sample_dir_path, save_file_name)
    draw_file = sitk.GetImageFromArray(draw_array)
    draw_file.CopyInformation(itk_file)
    sitk.WriteImage(draw_file, draw_file_path)
    print('saved the file : {}'.format(draw_file_path))
    pass

def check_mri_masking():
    print('start MRI label check.')
    sample_dir_path = '/home/soopil/Desktop/github/brainMRI_classification/sample_image/ADDlabel'
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
    in this case, we use
    aparc.DKTatlas+aseg.nii
    17,53 : left and right hippocampus label
    18,54 : amigdala label
    '''
    new_file = file_name[1] # aparc.DKTatlas+aseg.nii
    new_file_path = os.path.join(sample_dir_path, new_file)
    new_array, itk_file = read_MRI(new_file_path, label)
    new_label_list = count_label_num(new_array)

    dilation_iter = 2
    # extract label 17 area
    hippo_color = 2
    lh_hippo_pos = np.where(new_array == 17)
    empty_space_shape = [256 for i in range(3)]
    lh_hippo_label = np.zeros(empty_space_shape)
    lh_hippo_label[lh_hippo_pos] = hippo_color
    # dialation
    lh_hippo_label = ndimage.morphology.binary_dilation(lh_hippo_label, iterations=dilation_iter).astype(lh_hippo_label.dtype)
    lh_hippo_label[np.where(lh_hippo_label)] = hippo_color
    lh_hippo_label[lh_hippo_pos] = hippo_color + 1

    # extract label 18 area
    amig_color = 4
    lh_amig_pos = np.where(new_array == 18)
    empty_space_shape = [256 for i in range(3)]
    lh_amig_label = np.zeros(empty_space_shape)
    lh_amig_label[lh_amig_pos] = amig_color

    label_color = 1
    patch_size = 48

    lh_amig_label = ndimage.morphology.binary_dilation(lh_amig_label, iterations=dilation_iter).astype(
        lh_amig_label.dtype)
    lh_amig_label[np.where(lh_amig_label)] = amig_color
    lh_amig_label[lh_amig_pos] = amig_color + 1

    # bounding box drawing
    # center_pos = label_size_check(new_array, 17, isp)
    # draw_array = draw_patch_box(lh_hippo_label, center_pos, patch_size, label=label_color)
    draw_array = lh_amig_label + lh_hippo_label
    save_file_name = 'mask_lh_hippo_amig'+str(dilation_iter)+'.nii'
    save_nifti_file(draw_array, itk_file, sample_dir_path, save_file_name)

def check_mri_label():
    print('start MRI label check.')
    sample_dir_path = '/home/soopil/Desktop/github/brainMRI_classification/sample_image/ADDlabel'
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
    label_array, itk_file = read_MRI(new_file_path, label)
    new_label_list = count_label_num(label_array)
    print(new_label_list)
    lh_cort, rh_cort = [],[]
    subcort = []
    for label in sorted(new_label_list):
        if label // 1000 == 1:
            lh_cort.append(label)
        elif label // 1000 == 2:
            rh_cort.append(label)
        else:
            subcort.append(label)
    # print(lh_cort)
    # print(rh_cort)
    # assert False

    empty_space_shape = [256 for i in range(3)]
    empty_space = np.zeros(empty_space_shape)

    patch_size = 24
    label_color = 1 # red
    draw_array = empty_space

    """
    sub_cortical part box drawing part...
    """
    for lh_label in subcort:
        if lh_label == 91 or (lh_label >= 10 and lh_label <= 30) and (lh_label not in (14, 15, 16,24)):
            center_pos = label_size_check(label_array, lh_label, isp)
            draw_array[np.where(label_array == lh_label)] = lh_label
            draw_array = draw_patch_box(draw_array, center_pos, patch_size, label=label_color, thickness= 0)
    save_file_name = 'draw_patch_subcort_lh' + '.nii'
    save_nifti_file(draw_array, itk_file, sample_dir_path, save_file_name)
    assert False
    """
    cortical part box drawing part...
    """

    for lh_label in rh_cort:
        center_pos = label_size_check(label_array, lh_label, isp)
        draw_array[np.where(label_array == lh_label)] = lh_label
        draw_array = draw_patch_box(draw_array, center_pos, patch_size, label=label_color, thickness= 0)
    draw_array[np.where(label_array == 53)] = 53
    draw_array[np.where(label_array == 17)] = 17

    # for i in range(len(new_label_list)):
    #     if not new_label_list[i] in orig_label_list:
    #         print(new_label_list[i])
    """
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
    """

    save_file_name = 'draw_patch_cort_lh'+'.nii'
    save_nifti_file(draw_array, itk_file, sample_dir_path, save_file_name)
    # draw_file_path = os.path.join(sample_dir_path, 'draw_patch_only.nii')
    # draw_file = sitk.GetImageFromArray(draw_array)
    # draw_file.CopyInformation(itk_file)
    # sitk.WriteImage(draw_file, draw_file_path)
    # print('saved the file : {}'.format(draw_file_path))

def main():
    check_mri_label()
    # check_mri_masking()

if __name__ == '__main__':
    main()

