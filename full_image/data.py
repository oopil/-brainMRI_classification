import sys
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
sys.path.append('..')
sys.path.append('/home/soopil/Desktop/Dataset/github/brainMRI_classification/ConvNeuralNet')
from data_merge import *

def _read_py_func(path, label, is_aug = True, is_decode = True):
    isp = False
    if isp:print("file path : {}" .format(path))
    if is_decode:
        path_decoded = path.decode()
    else:
        path_decoded = path
    img_path_decoded, label_path_decoded = path_decoded.split(',')
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    # label_itk_file = sitk.ReadImage(label_path_decoded)
    # label_array = sitk.GetArrayFromImage(label_itk_file)
    shape = np.shape(array)
    hs = (shape[0] - shape[0]//3 ) // 2
    x,y,z = shape[0]//2, shape[1]//2, shape[2]//2,
    # print(hs*2)
    # print(x,y,z)
    # transition augmentation
    if is_aug:
        ran = np.random.randint(10, size=(3))
        x += ran[0]
        y += ran[1]
        z += ran[2]
    # print(x,y,z)
    image_cropped = array[x - hs:x + hs, y - hs:y + hs, z - hs:z + hs]
    image_cropped = np.expand_dims(image_cropped, 3)
    if is_decode:
        return image_cropped.astype(np.float32), label.astype(np.int32)
    else:
        return image_cropped.astype(np.float32), label

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
    dataset = tf.data.Dataset.from_tensor_slices((tr_x,tr_y))
    dataset = dataset.map(
        lambda tr_x, tr_y:
        tuple(tf.py_func(_read_py_func, [tr_x, tr_y], [tf.float32, tf.int32])),
        num_parallel_calls=5)

    print(dataset.output_types)
    print(dataset.output_shapes)
    print(dataset)
    # dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    # define iterations
    iterations = len(tr_y) // batch_size
    if (len(tr_y) % batch_size) != 0:
        iterations -= 1
    # dataset = dataset.range(iterations)

    # img_size = 192
    # handle = tf.placeholder(tf.string, shape=[])
    # iterator = tf.data.Iterator.from_string_handle(handle, dataset.output_types,([None,img_size,img_size,img_size,1],[None]))

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element, iterator,iterations # handle,

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
    elif sv_set == 185:
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

if __name__ == '__main__':
    whole_set = read_cnn_data(0)
    tr_x, tr_y, tst_x, tst_y = whole_set[1]
    _read_py_func(tr_x[0], tr_y[0], is_decode=False)
    # sess = tf.Session()

