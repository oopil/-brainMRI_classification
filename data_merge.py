from sklearn.utils import shuffle
from class_metabot import *
import openpyxl
import numpy as np

class MRI_chosun_data():
    def __init__(self):
        self.class_array = []
        self.diag_type = "clinic" # or "new" or "PET"
        self.clinic_diag_index = 5
        self.new_diag_index=6
        self.pet_diag_index=3 # ???

        self.option = ['P', 'T', 'V']
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
        self.label_info_list = \
            [[self.data_excel[i][0],self.data_excel[i][4],self.data_excel[i][5],self.data_excel[i][6]]\
             for i in range(1, len(self.data_excel)) if i%3 == 0]
        print('label infomation length : {}' .format(len(self.label_info_list)))
        return self.label_info_list

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

    def merge_path_and_label(self):
        for i in range(10):
            print(self.input_image_path_list[i][1], self.label_info_list[i])
        pass

        for i, path in enumerate(self.input_image_path_list):
            id = path[1]
            excel_index = self.label_info_list.index(id)

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

    def is_all_zero(self, l:list, idx:int)->bool:
        for e in l:
            if e[idx]:
                return False
        return True

    def is_male(self, gender:str):
        if gender == 'M': return True
        elif gender == 'F': return False
        else :
            print('wrong sex is detected.')
            print(gender)
            assert False


def test_something_2():
    loader = MRI_chosun_data()
    base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    loader.read_excel_data(excel_path)
    path_list = loader.extr_input_path_list(base_folder_path)
    print(path_list[0])
    loader.get_label_info_excel()
    loader.merge_path_and_label()
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

def dataloader(class_option : str, option_num, is_merge=False):
    loader = MRI_chosun_data()
    data = loader.read_excel_data()
    return loader.extr_data(data, is_merge, class_option, option_num)




if __name__ == '__main__':
    test_something_2()
    assert False

'''
<<numpy array column api>> 
self.data_excel[:,0]
'''