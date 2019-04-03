from sklearn.utils import shuffle
from class_metabot import *
from decorator import *
import openpyxl
import numpy as np

class MRI_chosun_data():
    def __init__(self):
        self.class_array = []
        self.nn_data = []
        self.nn_label = []
        self.cnn_data = []
        self.cnn_label = []
        self.diag_type = "None" # or "new" or "PET"
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
            # 'NC vs AD': ['CN', 'AD'],
            # 'NC vs MCI': ['CN', 'MCI'],
            # 'MCI vs AD': ['MCI', 'AD'],
            # 'NC vs MCI vs AD': ['CN', 'MCI', 'AD'],
            'NC vs mAD vs aAD vs AD': ['NC', 'mAD', 'aAD', 'ADD']
        }

        self.class_option_dict_pet = {
            'pos vs neg': ['positive', 'negative']
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
        print(excel_option, option_index)
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
        print(l1, l2)
        print('remove zero value only columns.')
        for col in range(l2):
            is_zero_col = True
            col_index = l2 - col - 1
            for row in range(l1):
                print(type(self.nn_data[row][4]))
                if self.nn_data[row][col_index]:
                    print(self.nn_data[row][4])
                    # print(self.nn_data[row][col_index])
                    is_zero_col = False
                    break

            if is_zero_col:
                print('delete column.')
                delete_col_count += 1
                self.nn_data = np.delete(self.nn_data, col_index, 1)
            # assert False
        print('removed {} columns.'.format(delete_col_count))
        print(len(self.nn_data), len(self.nn_data[0]))

        return self.nn_data

    def divide_data_and_label(self):
        self.nn_data = np.array(self.nn_data)
        print(type(self.nn_data[0][4]), self.nn_data[0][4])
        print('divide the nn_data into nn_data and label.')
        orig_length = len(self.nn_data[0])
        self.nn_label = self.nn_data[:,:4]
        self.nn_data = self.nn_data[:,4:]
        print(orig_length, len(self.nn_label), len(self.nn_data))

    def define_label(self, label_info):
        pass

    def extr_nn_data(self):
        label_info = self.cnn_data[excel_index][1:4]  # beta amyloid / new / clinic diagnosis
        label = self.define_label(label_info)
        self.cnn_data[excel_index].append(label)

    def extr_cnn_data(self):
        pass



    def get_class_name(self, l:list, idx:int) -> list:
        temp = []
        index = 0
        print(len(l))
        for e in l:
            index  += 1
            if index == 1:
                continue
            temp.append(e[idx])

        temp = list(set(temp))
        print('get names of class')
        print(temp)
        return temp

    def count_col_data(self, l:list, type:str, index:int) -> None:
        count = 0
        for e in l:
            if e[index] == type:
                count += 1
        print('it has ', int(count/3), type, 's.')

    def get_class_array(self, class_option):
        if self.diag_type == "clinic":
            return self.class_option_dict_clinic[class_option]

    def extr_data(self, data, is_merge, class_option, option_num=0) :
        self.class_array = self.get_class_array(class_option)
        class_num = len(self.class_array)
        print('remove some data to use it ... ')
        remove_idx_l = [0,1,4,5,6,7]
        if self.diag_type == "clinic":
            opt_dict = self.opt_dict_clinic
            class_index = 5
        elif self.diag_type == "new":
            opt_dict = self.opt_dict_new
            class_index = 6
        # print(data[0])
        data.remove(data[0]) # remove feature line
        # print(data[0])

        option = option_num  # P T V options
        new_data = []
        new_label = []
        if is_merge:
            length = len(data)
            assert length % 3 == 0
            for i in range(length//3):
                # extract only one option features among P, V, T options
                # remove MCI instances

                label = self.get_class(class_num, data[i*3][class_index])
                if label == -1:
                    continue
                    # label.append(self.get_class(class_num, data[i][class_index]))
                new_element = []
                for option in range(3): # from the option features P V T
                    for j in range(len(data[i*3+option])):
                        if j in remove_idx_l:
                            continue
                        new_element.append(data[i*3+option][j])
                    # print(len(new_element))

                new_data.append(new_element)
                new_label.append(label)
            pass
        else:
            for i in range(len(data)):
                # extract only one option features among P, V, T options
                # remove MCI instances
                label = self.get_class(class_num, data[i][class_index])
                if i % 3 != option or label == -1:
                    continue
                    # label.append(self.get_class(class_num, data[i][class_index]))

                new_element = []
                for j in range(len(data[i])):
                    if j in remove_idx_l:
                        continue
                    new_element.append(data[i][j])

                new_data.append(new_element)
                new_label.append(label)
        # print(new_data[0])
        # print(label)
        print(len(new_data), len(new_label))
        # print(len(new_data[0]), new_data[0])
        print(len(new_data[0]))
        return new_data, new_label

    def get_class(self, class_num, class_name) -> int:
        if self.diag_type == "clinic":
            pass
        elif self.diag_type == "new":
            print('this code is not prepared yet.')
            assert False

        for i, c in enumerate(self.class_array):
            if c in class_name:
                return i

        if 'MCI' in class_name or 'AD' in class_name or 'CN' in class_name:
            return -1

        print('AD' in class_name)
        print(self.class_array , class_name)
        assert False

    def get_class_3d(self, class_option, class_name) -> int:
        class_array = self.get_class_array(class_option)
        if self.diag_type == "clinic":
            for i, c in enumerate(class_array):
                if c in class_name:
                    return i

            if 'MCI' in class_name or 'AD' in class_name or 'CN' in class_name:
                return -1

            print('inappropriate class name : ')
            print(self.diag_type, class_array, class_name)
            assert False
            pass

        elif self.diag_type == "new":
            print('this code is not prepared yet.')
            assert False

        elif self.diag_type == "PET":
            assert False

    def shuffle(self):
        pass

    def split_train_and_test(self):
        pass

    def is_all_zero(self, l:list, idx:int)->bool:
        for e in l:
            if e[idx]:
                return False
        return True

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

def dataloader(class_option : str, option_num, is_merge=False):
    loader = MRI_chosun_data()
    data = loader.read_excel_data()
    return loader.extr_data(data, is_merge, class_option, option_num)

def NN_dataloader():
    '''
    1. read excel data
    2. squeeze 3 lines into 1 lines according to the options P V T
    3. make label list
    3. remove zero value only column
    4. shuffle
    5. normalization
    6. split train and test dataset
    :return: train and test data and lable
    '''
    loader = MRI_chosun_data()
    loader.set_diagnosis_type('new')
    base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'

    excel_option = 'P' # P V T merge
    loader.read_excel_data(excel_path)
    loader.squeeze_excel(excel_option=excel_option)
    loader.remove_zero_column()
    loader.define_label()
    pass

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
#     ford_num = 3
#     ford_index = 0
#     data, label = dataloader(class_option[class_option_index], option_num, is_merge=is_merge)
#     # assert False
#     data, label = shuffle_two_arrays(data, label)
#     X_train, Y_train, X_test, Y_test = split_train_test(data, label, ford_num, ford_index)
#     # print(len(data[0]), len(X_train[0]))
#     # X_train, Y_train = valence_class(X_train, Y_train, class_num)
#     # X_test, Y_test = valence_class(X_test, Y_test, class_num)
#     print(len(Y_train))
#     X_train, Y_train = over_sampling(X_train, Y_train, sampling_option)
#     print(len(Y_train))
#
#     train_num = len(Y_train)
#     test_num = len(Y_test)
#     feature_num = len(X_train[0])
#     print(X_train.shape, X_test.shape)

'''
<<numpy array column api>> 
self.data_excel[:,0]
'''