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

        over_cover = np.where(gap_array == lh_array)
        gap_array[np.where(gap_array == lh_array)] = 0
        gap_array[np.where(gap_array == rh_array)] = 0

        # for i in over_cover:
        #     gap_array[i] = 0

        print(over_cover)
        gap_array = normalize_3D(gap_array, 256)
        print(gap_array.shape)
        # assert False

        lh_file = sitk.GetImageFromArray(lh_array)
        rh_file = sitk.GetImageFromArray(rh_array)
        gap_file = sitk.GetImageFromArray(gap_array)
        lh_file.CopyInformation(itk_file)
        rh_file.CopyInformation(itk_file)
        gap_file.CopyInformation(itk_file)

        # assert False
        # sitk.WriteImage(lh_file, lh_file_path)
        # sitk.WriteImage(rh_file, rh_file_path)
        sitk.WriteImage(gap_file, gap_file_path)

def main():
    split_brain_mri()

if __name__ == '__main__':
    main()

