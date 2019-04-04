from sklearn.utils import shuffle
from class_metabot import *
from decorator import *
from random import shuffle
import openpyxl
import numpy as np

class MRI_chosun_data():
    def __init__(self):
        self.class_array = []
        self.nn_data = []
        self.nn_label = []
        self.cnn_data = []
        self.cnn_label = []
        self.diag_type = "None" # or "clinic" or "new" or "PET"
        self.clinic_diag_index = 5
        self.new_diag_index=6
        self.pet_diag_index=3 # ???

        self.excel_option = ['P', 'T', 'V', 'merge']
        self.opt_dict_clinic = {
            'AD': 0,
            'CN': 1,
            'aMCI': 2,
            'naMCI': 3
        }

        self.opt_dict_new = {
            'aAD': 0,
            'NC': 1,
            'ADD': 2,
            'mAD': 3
        }

        self.class_option_dict_clinic = {
            'NC vs AD': ['CN', 'AD'],
            'NC vs MCI': ['CN', 'MCI'],
            'MCI vs AD': ['MCI', 'AD'],
            'NC vs MCI vs AD': ['CN', 'MCI', 'AD']
        }

        self.class_option_dict_new = {
            'NC vs ADD': ['NC', 'ADD'],
            'NC vs mAD vs aAD vs ADD': ['NC', 'mAD', 'aAD', 'ADD']
        }

        self.class_option_dict_pet = {
            'PET pos vs neg': ['positive', 'negative']
        }
#%%
    def set_diagnosis_type(self, type):
        self.diag_type = type
#%%
    def read_excel_data(self, excel_path):
        xl_file_name = excel_path
        xl_password = '!adai2018@#'
        xl = openpyxl.load_workbook(xl_file_name, read_only=True)
        ws = xl['Sheet1']
        self.data_excel = []
        for row in ws.rows:
            line = []
            for cell in row:
                line.append(cell.value)
            self.data_excel.append(line)
        # self.data_excel = np.array(self.data_excel)
        return self.data_excel

    def get_label_info_excel(self):
        print('Column name : ')
        print(self.data_excel[0])
        index_list = [4,5,6] # PET NEW CLINIC
        '''
        ['MRI_id', 'gender', 'age', 'education', 'amyloid PET result', 'Clinic Diagnosis', 'New Diag',
        'mtype', 'c4', ...]
        '''
        self.cnn_data = \
            [[self.data_excel[i][0],self.data_excel[i][4],self.data_excel[i][5],self.data_excel[i][6]]\
             for i in range(1, len(self.data_excel)) if i%3 == 0]
        print('label infomation length : {}' .format(len(self.cnn_data)))
        return self.cnn_data

    def extr_input_path_list(self, base_folder_path):
        folder_name = ['aAD', 'ADD', 'mAD', 'NC']
        print('start to extract meta data from dataset folder')
        bot = MetaBot(base_folder_path)
        self.input_image_path_list = []
        for class_name in folder_name:
            self.input_image_path_list = self.input_image_path_list + bot.MRI_chosun(class_name)
        print(folder_name, len(self.input_image_path_list), self.input_image_path_list)
        del bot
        return self.input_image_path_list

    def merge_info(self):
        '''
        merge the real data path and excel label information.
        '''
        excel_np =  np.array(self.cnn_data)
        excel_id_col = list(excel_np[:,0])
        for i, path in enumerate(self.input_image_path_list):
            '''
            self.input_image_path_list is aligned with the order [aAD, ADD, mAD NC]
            '''
            id ,input_path = path[1], path[2]
            excel_index = excel_id_col.index(id)
            self.cnn_data[excel_index].append(input_path)

        # for i in range(10):
        #     print(self.cnn_data[i])

    def squeeze_excel(self, excel_option):
        '''
        because there are 3 line for each patient basically,
        we have to choose how to handle it.

        choose only one line or merge all of them
        and then remove only zero column
        :param option:
        :return:
        '''
        print('squeeze the excel.')
        if not excel_option in self.excel_option:
            print('the excel option in not proper.')
            print(excel_option, self.excel_option)
            assert False

        option_index = self.excel_option.index(excel_option)
        print('excel option : ',excel_option, option_index)
        for i in range(1,len(self.data_excel)):

            # print(label_info)
            if (i-1)%3 == option_index:
                line = self.data_excel[i][8:]
                label_info = self.data_excel[i][4:7]
                new_line = line
                self.nn_data.append(new_line)
                self.nn_label.append(label_info)
                # print(len(self.data_excel[i]), len(line))
                # print(new_line)

            if option_index == 3 and i%3 == 1:
                line = [self.data_excel[i+k][8:] for k in range(3)]
                label_info = self.data_excel[i][4:7]
                new_line = line[0] + line[1] + line[2]
                self.nn_data.append(new_line)
                self.nn_label.append(label_info)
                # print(len(self.data_excel[i][:10]), len(line[0]), len(line[1]), len(line[2]))
                # print(new_line)
        return self.nn_data, self.nn_label

    def remove_zero_column(self):
        self.nn_data = np.array(self.nn_data)
        l1, l2 = len(self.nn_data), len(self.nn_data[0])
        delete_col_count = 0
        print('remove zero value only columns.')
        print('matrix size',l1, 'X', l2)
        for col in range(l2):
            is_zero_col = True
            col_index = l2 - col - 1
            for row in range(l1):
                # print(type(self.nn_data[row][4]))
                if self.nn_data[row][col_index]:
                    # print(self.nn_data[row][4])
                    # print(self.nn_data[row][col_index])
                    is_zero_col = False
                    break

            if is_zero_col:
                # print('delete column.')
                delete_col_count += 1
                self.nn_data = np.delete(self.nn_data, col_index, 1)
            # assert False
        print('removed {} columns.'.format(delete_col_count))
        print('{:<5}=>{:<5}' .format(l2, len(self.nn_data[0])) )
        return self.nn_data, self.nn_label

#%%
    def label_pet(self, label, class_array):
        for i, c in enumerate(self.class_array):
            if c in label:
                return i
                # print(c, label)
                break
        print('there is no appropriate label name. :')
        print(self.class_array, label)
        assert False

    def label_clinic(self, label, class_array):
        for i, c in enumerate(self.class_array):
            if c in label:
                '''
                print(label, i) # check the labeling state                
                '''
                return i
        if 'MCI' in label or 'AD' in label or 'CN' in label:
            return -1
        print('there is no appropriate label name. :')
        print(self.class_array, label)
        assert False

    def label_new(self, label, class_array):
        for i, c in enumerate(self.class_array):
            if c in label:
                return i
        if 'NC' in label or 'AD' in label:
            return -1
        print('there is no appropriate label name. :')
        print(self.class_array, label)
        assert False

    def define_label(self, label_info, class_option):
        '''
        :param label_info: it has 3 columns : pet, new, clinic
        :param class_option:  'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
        :return:

        when we use the class option as NC vs AD, we need to remove MCI line.
        '''
        is_print = False
        if is_print: pass
        print('start labeling...' )
        label_info = np.array(label_info)
        self.label_list = []
        # self.class_array = self.get_class_array(class_option)
        if self.diag_type == 'PET':
            self.class_array = self.class_option_dict_pet[class_option]
            label_name = label_info[:,0]
            # print(label_name, self.class_array)
            for i, label in enumerate(label_name):
                self.label_list.append(self.label_pet(label, self.class_array))

        elif self.diag_type == 'clinic':
            self.class_array = self.class_option_dict_clinic[class_option]
            label_name = label_info[:,1]
            for i, label in enumerate(label_name):
                self.label_list.append(self.label_clinic(label, self.class_array))

        elif self.diag_type == 'new':
            self.class_array = self.class_option_dict_new[class_option]
            label_name = label_info[:,2]
            for i, label in enumerate(label_name):
                self.label_list.append(self.label_new(label, self.class_array))

        else:
            print('diagnosis type is wrong. : ', self.diag_type)
            assert False
        print('class option : {} / class array : {}'.format(class_option, self.class_array))

        '''
        remove the -1 line of label and data
        '''
        print('remove the -1 line of label and data.')
        self.label_list = np.array(self.label_list)

        if is_print:
            print(self.label_list)
            print(len(self.nn_data),len(self.label_list))
        label_length = len(self.label_list)
        for row in range(len(self.label_list)):
            row_index = label_length - row - 1
            label = self.label_list[row_index]
            # print(row, len(self.label_list), row_index)
            if label == -1:
                self.nn_data = np.delete(self.nn_data, row_index, 0)
                self.label_list = np.delete(self.label_list, row_index, 0)
            pass

        if is_print:
            print(len(self.nn_data),len(self.label_list))
            print(type(self.nn_data),type(self.label_list))
            # print(self.nn_data[0])
            print(self.label_list)

        return self.nn_data, self.label_list

    def shuffle_data(self, data, label):
        assert len(data)==len(label)
        random_list = [i for i in range(len(data))]
        '''
        choose between shuffle and shuffle_static
        # random_list = shuffle_static( ... )
        '''
        shuffle(random_list)
        # print(random_list)
        self.shuffle_data, self.shuffle_label = [],[]
        for index in random_list:
            self.shuffle_data.append(data[index])
            self.shuffle_label.append(label[index])

        assert len(label) == len(self.shuffle_label)
        return self.shuffle_data, self.shuffle_label

    def split_data_by_num(self, data, label, test_num):
        '''
        :param data:
        :param label:
        :param test_num:
        :return:just one train and test set.
        '''
        label_set = list(set(label))
        print('split the data into train and test by test number. test number : {} label set :{}'\
              .format(test_num, label_set))
        # print(type(label_set))
        label_count = [0 for _ in range(len(label_set))]
        self.test_data, self.test_label = [], []
        self.train_data, self.train_label = [], []
        for i, l in enumerate(label):
            # print(i,l,label_count)
            if label_count[l] < test_num:
                label_count[l] += 1
                self.test_data.append(data[i])
                self.test_label.append(label[i])
                continue

            self.train_data.append(data[i])
            self.train_label.append(label[i])

        print('train data : {} / train label : {} / test data : {} / test label : {}'\
            .format(len(self.train_data), len(self.train_label), len(self.test_data), len(self.test_label)))
        return self.train_data, self.train_label, \
               self.test_data, self.test_label

    def split_data_by_fold(self, data, label, fold_num):
        '''
        :param data:
        :param label:
        :param fold_num:
        :return: return all possible train and test set according to the fold number.
        '''

        label_set = list(set(label))
        print('split the data into train and test by fold number. fold number : {} label set :{}' \
              .format(fold_num, label_set))
        separate_data = [[] for _ in range(len(label_set))]
        separate_label = [[] for _ in range(len(label_set))]
        # separate the data into different label list
        for i, l in enumerate(label):
            # print(i,l,label_count)
            separate_data[l].append(data[i])
            separate_label[l].append(label[i])
        print(separate_label)
        label_count = [len(i) for i in separate_label]
        test_count = [count//fold_num for count in label_count]
        print(label_count, test_count)
        smaller_data_num = min(label_count)
        test_num = smaller_data_num//fold_num

        whole_set = []
        for fold_index in range(fold_num):
            train_data, train_label, test_data, test_label = [], [], [], []
            for i, one_label in enumerate(separate_data):
                if fold_index == fold_num - 1:
                    train_data = train_data + one_label[:test_count[i] * fold_index]
                    train_label = train_label + separate_label[i][:test_count[i] * fold_index]
                    test_data = test_data + one_label[test_count[i] * fold_index:]
                    test_label = test_label + separate_label[i][test_count[i] * fold_index:]
                    pass
                else:
                    train_data = train_data + one_label[:test_count[i] * fold_index] + one_label[test_count[i] * (fold_index + 1):]
                    train_label = train_label + separate_label[i][:test_count[i] * fold_index] + separate_label[i][test_count[i] * (fold_index + 1):]
                    test_data = test_data + one_label[test_count[i] * fold_index:test_count[i] * (fold_index + 1)]
                    test_label = test_label + separate_label[i][test_count[i] * fold_index:test_count[i] * (fold_index + 1)]
            # print(train_label)
            # print(test_label)
            print(len(train_label)+len(test_label), len(train_label), len(test_label))
            whole_set.append([train_data, train_label, test_data, test_label])
        return whole_set

#%%
from imblearn.over_sampling import *
from imblearn.combine import *
def over_sampling(X_imb, Y_imb, sampling_option):
    print('starts over sampling ...', sampling_option)
    if sampling_option == 'ADASYN':
        X_samp, Y_samp = ADASYN(random_state=0).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'SMOTE':
        X_samp, Y_samp = SMOTE(random_state=4).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'SMOTEENN':
        X_samp, Y_samp = SMOTEENN(random_state=0).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'SMOTETomek':
        X_samp, Y_samp = SMOTETomek(random_state=4).fit_sample(X_imb, Y_imb)
    elif sampling_option == 'None':
        X_samp, Y_samp = X_imb, Y_imb
    else:
        print('sampling option is not proper.', sampling_option)
        assert False
    imbalance_num = len(Y_imb)
    balance_num = len(Y_samp)
    print('over sampling from {:5} -> {:5}.'.format(imbalance_num, balance_num))
    return X_samp, Y_samp

def shuffle_static(arr1, arr2):
    return shuffle(arr1, arr2, random_state=0)

def valence_class(data, label, class_num):
    print('Valence the number of train and test dataset')
    length = len(data)
    label_count = [0 for i in range(class_num)]
    label_count_new = [0 for i in range(class_num)]

    for i in sorted(label):
        label_count[i] += 1

    # print('label count : ', label_count)
    min_count = min(label_count)
    print(min_count)
    new_data = []
    new_label = []
    for i, k in enumerate(label):
        if label_count_new[k] > min_count:
            continue
        new_data.append(data[i])
        new_label.append(label[i])
        label_count_new[k] += 1
    # print('new label count : ', label_count_new)
    print('down sampling from {} -> {}.'.format(label_count, label_count_new))
    return np.array(new_data), np.array(new_label)

@datetime_decorator
def test_something_2():
    loader = MRI_chosun_data()
    loader.set_diagnosis_type('new')
    base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    loader.read_excel_data(excel_path)
    path_list = loader.extr_input_path_list(base_folder_path)
    print(path_list[0])
    loader.get_label_info_excel()
    loader.merge_info()

def normalize(X_):
    return (X_-X_.min(0))/X_.max(axis=0)

def NN_dataloader(diag_type, class_option, base_folder_path, \
                  excel_path, excel_option, test_num, fold_num, is_split_by_num):
    '''
    1. read excel data (O)
    2. squeeze 3 lines into 1 lines according to the options P V T merge (O)
    3. remove zero value only column (O)
    3. make label list (O)
    4. shuffle (O)
    5. normalization
    6. split train and test dataset (O)
    :return: train and test data and lable
    '''

    # "clinic" or "new" or "PET"
    # 'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
    # diag_type = "PET"
    # class_option = 'PET pos vs neg'
    # diag_type = "new"
    # class_option = 'NC vs mAD vs aAD vs ADD'
    # diag_type = "clinic"
    # class_option = 'NC vs AD' #'NC vs MCI vs AD'

    loader = MRI_chosun_data()
    loader.set_diagnosis_type(diag_type)
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    # excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'

    # excel_option = 'merge' # P V T merge
    loader.read_excel_data(excel_path)
    loader.squeeze_excel(excel_option=excel_option)
    data, label_info = loader.remove_zero_column()
    data, label = loader.define_label(label_info, class_option)
    data = normalize(data)
    '''
    when split the data by fold number, should we split the data earlier than shuffle??
    '''
    # is_split_by_num = False
    if is_split_by_num:
        shuffle_data, shuffle_label = loader.shuffle_data(data, label)
        # test_num = 20
        '''
            return only one train and test set
        '''
        return loader.split_data_by_num(shuffle_data, shuffle_label, test_num)
    else:
        shuffle_data, shuffle_label = loader.shuffle_data(data, label)
        # fold_num = 5
        # fold_index = 0
        '''
            return all train and test sets devided by fold. 
        '''
        return loader.split_data_by_fold(shuffle_data, shuffle_label, fold_num)

def CNN_dataloader():
    '''
    1. read excel data
    2. read input file path from dataset folder path
    3. merge excel and path information
    4. make label list
    5. shuffle
    5. normalization
    7. split train and test dataset
    :return: train and test data and lable
    '''
    loader = MRI_chosun_data()
    loader.set_diagnosis_type('new')
    base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    loader.read_excel_data(excel_path)
    path_list = loader.extr_input_path_list(base_folder_path)
    print(path_list[0])
    loader.get_label_info_excel()
    loader.merge_info()
    pass

if __name__ == '__main__':
    NN_dataloader()
    # test_something_2()
    assert False

#
# def test_something():
#     is_merge = True  # True
#     option_num = 0  # P V T options
#     '''
#     I should set the class options like
#     NC vs AD
#     NC vs MCI
#     MCI vs AD
#
#     NC vs MCI vs AD
#     '''
#     class_option = ['NC vs AD', 'NC vs MCI', 'MCI vs AD', 'NC vs MCI vs AD']
#     class_option_index = 3
#     class_num = class_option_index // 3 + 2
#     sampling_option = 'ADASYN'
#     fold_num = 3
#     fold_index = 0
#     data, label = dataloader(class_option[class_option_index], option_num, is_merge=is_merge)
#     # assert False
#     data, label = shuffle_two_arrays(data, label)
#     train_data, train_label, test_data, test_label = split_train_test(data, label, fold_num, fold_index)
#     # print(len(data[0]), len(train_data[0]))
#     # train_data, train_label = valence_class(train_data, train_label, class_num)
#     # test_data, test_label = valence_class(test_data, test_label, class_num)
#     print(len(train_label))
#     train_data, train_label = over_sampling(train_data, train_label, sampling_option)
#     print(len(train_label))
#
#     train_num = len(train_label)
#     test_num = len(test_label)
#     feature_num = len(train_data[0])
#     print(train_data.shape, test_data.shape)

'''
<<numpy array column api>> 
self.data_excel[:,0]
'''