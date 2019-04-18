import os
import sys
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification')
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification/sample_image')
import numpy as np
import SimpleITK as sitk

def normalize_3D(X_, L):
    min = np.amin(X_)
    max = np.amax(X_)
    if max == 0:
        print('failed to normalize. the max element is zero.')
        print(min)
        print(max)
        # print(X_)
        assert False
    print(min)
    print(max)
    return (L-1)*(X_-min)/(max-min)

def split_brain_mri():
    sample_folder_path = '/home/sp/PycharmProjects/brainMRI_classification/sample_image'
    # file_name = ['brain_aAD.nii','brain_ADD.nii']
    file_name = ['brain_NC.nii']
    for file in file_name:
        file_path = os.path.join(sample_folder_path, file)
        # file_path = file
        file_name = file.split('.')[0]
        lh_file_path = os.path.join(sample_folder_path, file_name + '_lh.nii')
        rh_file_path = os.path.join(sample_folder_path, file_name + '_rh.nii')
        gap_file_path = os.path.join(sample_folder_path, file_name + '_gap.nii')
        print(file_path)
        itk_file = sitk.ReadImage(file_path)
        array = sitk.GetArrayFromImage(itk_file)
        print(type(array), array.shape)

        lh_rh_axis = 2
        empty_space_shape = [256 for i in range(3)]
        empty_space_shape[lh_rh_axis] = 128
        empty_space = np.zeros(empty_space_shape)
        print(empty_space_shape)

        lh_array = np.concatenate((array[:,:,:128], empty_space), axis=lh_rh_axis)
        rh_array = np.flip(array[:,:,128:], axis=2)
        # rh_array = array[:,:,128:]
        rh_array = np.concatenate((rh_array,empty_space), axis=lh_rh_axis)
        gap_array = np.subtract(lh_array,rh_array)
        gap_array = np.abs(gap_array)

        gap_array = normalize_3D(gap_array, 256)
        print(gap_array.shape)
        # assert False

        lh_file = sitk.GetImageFromArray(lh_array)
        rh_file = sitk.GetImageFromArray(rh_array)
        gap_file = sitk.GetImageFromArray(gap_array)
        lh_file.CopyInformation(itk_file)
        rh_file.CopyInformation(itk_file)
        gap_file.CopyInformation(itk_file)
        sitk.WriteImage(lh_file, lh_file_path)
        sitk.WriteImage(rh_file, rh_file_path)
        sitk.WriteImage(gap_file, gap_file_path)

def load_nii_data(path):
    result_path = 'eq_' + path
    itk_file = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(itk_file)
    array_eq = exposure.equalize_hist(array)
    min_intensity = array_eq.min(axis=(0,1,2), keepdims=True)
    max_intensity = array_eq.max(axis=(0, 1, 2), keepdims=True)

    array_eq_normal = array_eq *(max_intensity/(max_intensity-min_intensity))
    print(min_intensity, max_intensity)
    shape_arr = array.shape

    slice1 = int(shape_arr[0]/2)
    slice2 = int(shape_arr[1]/ 2)
    slice3 = int(shape_arr[2]/ 2)
    print(array[slice1:slice1+1,slice2:slice2+1, slice3:slice3+5])
    print(array_eq[slice1:slice1 + 1, slice2:slice2 + 1, slice3:slice3+5])
    print(array_eq_normal[slice1:slice1+1,slice2:slice2+1, slice3:slice3+5])
    print()
    new_file = sitk.GetImageFromArray(array_eq)
    new_file.CopyInformation(itk_file)
    sitk.WriteImage(new_file, result_path)

    # img = nib.load(path)
    # new_header = header = img.header.copy()
    # result_path = 'equalization_' + path
    # img_array = img.get_fdata()
    # img_eq = exposure.equalize_hist(img_array)
    # print(img.affine)
    # print(array)
    # nib.save(img_eq, result_path)
    return


def main():
    split_brain_mri()

if __name__ == '__main__':
    main()

