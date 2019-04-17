import SimpleITK as sitk
# import nibabel as nib
from skimage import exposure
file_path = ['T1.nii.gz','T1Brain.nii.gz']
file_path = ['T1_NC.nii.gz','T1Brain_NC.nii.gz']

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
    for file in file_path:
        load_nii_data(file)

if __name__ == '__main__':
    main()

