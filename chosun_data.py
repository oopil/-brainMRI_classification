from excel_class import XL, Printer
from sklearn.utils import shuffle
import openpyxl
import numpy as np
import random

class DataLoader():
    def __init__(self, frd):
        self.meta = []
        self.class_array = []
        self.diag_type = "clinic" # or "new" or "PET"
        self.clinic_diag_index = 5
        self.new_diag_index=6

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

        self.opt_dict_PET = {
            'negative': 0,
            'positive': 1
        }

        self.class_option_dict_clinic = {
            'NC vs AD': ['CN', 'AD'],
            'NC vs MCI': ['CN', 'MCI'],
            'MCI vs AD': ['MCI', 'AD'],
            'NC vs MCI vs AD': ['CN', 'MCI', 'AD']
        }

        self.class_option_dict_new = {
            'NC vs AD': ['CN', 'AD'],
            'NC vs MCI': ['CN', 'MCI'],
            'MCI vs AD': ['MCI', 'AD'],
            'NC vs MCI vs AD': ['CN', 'MCI', 'AD']
        }

    def data_read(self):
        xl_file_name = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
        xl_password = '!adai2018@#'
        xl = openpyxl.load_workbook(xl_file_name, read_only=True)
        ws = xl['Sheet1']
        data = []
        for row in ws.rows:
            line = []
            for cell in row:
                line.append(cell.value)
            data.append(line)
        # print(len(data), len(data[0]))
        # print(data[0])
        # print(data[1])
        return data

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
    else :
        print('sampling option is not proper.', sampling_option)
        assert False
    imbalance_num = len(Y_imb)
    balance_num = len(Y_samp)
    print('over sampling from {:5} -> {:5}.'.format(imbalance_num, balance_num))
    return X_samp, Y_samp

def test_something():
    is_merge = True  # True
    option_num = 0  # P V T options
    '''
    I should set the class options like
    NC vs AD
    NC vs MCI
    MCI vs AD

    NC vs MCI vs AD
    '''
    class_option = ['NC vs AD', 'NC vs MCI', 'MCI vs AD', 'NC vs MCI vs AD']
    class_option_index = 3
    class_num = class_option_index // 3 + 2
    sampling_option = 'ADASYN'
    ford_num = 3
    ford_index = 0
    data, label = dataloader(class_option[class_option_index], option_num, is_merge=is_merge)
    # assert False
    data, label = shuffle_two_arrays(data, label)
    X_train, Y_train, X_test, Y_test = split_train_test(data, label, ford_num, ford_index)
    # print(len(data[0]), len(X_train[0]))
    # X_train, Y_train = valence_class(X_train, Y_train, class_num)
    # X_test, Y_test = valence_class(X_test, Y_test, class_num)
    print(len(Y_train))
    X_train, Y_train = over_sampling(X_train, Y_train, sampling_option)
    print(len(Y_train))

    train_num = len(Y_train)
    test_num = len(Y_test)
    feature_num = len(X_train[0])
    print(X_train.shape, X_test.shape)

def dataloader(class_option : str, option_num, is_merge=False):
    loader = DataLoader(2)
    data = loader.data_read()
    return loader.extr_data(data, is_merge, class_option, option_num)

def pathloader(option_num):
    loader = DataLoader(2)
    data = loader.data_read()
    print(data)

def shuffle_two_arrays(arr1, arr2):
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

def split_train_test(data, label, ford_num, ford_index):
    print('split the dataset into train and test sets.')
    sample_num = len(data)
    test_num = sample_num // ford_num
    train_num = sample_num - test_num
    # X = np.array(get_random_sample(X, sample_num))
    # Y = np.array(get_random_sample(Y, sample_num))
    X_ = np.array(data)
    Y_ = np.array(label)
    l1 , l2 = len(X_), len(X_[0])

    delete_col_count = 0
    print(l1, l2)
    print('remove 0 value columns.')
    for i in range(l2):
        col_index = l2 - i - 1
        for j in range(l1):
            if X_[j][col_index]:
                break
            # print('delete column.')
            delete_col_count += 1
            X_ = np.delete(X_, col_index, 1)
    print('removed {} columns.'.format(delete_col_count))
    print(len(data[0]), len(X_[0]))

    X_ = normalize(X_)
    # X_, Y_ = shuffle_two_arrays(X_, Y_)

    # assert ford_index < ford_num-1
    if ford_index == ford_num-1:
        X_train = X_[:test_num * ford_index]
        Y_train = Y_[:test_num * ford_index]
        X_test = X_[test_num * ford_index:]
        Y_test = Y_[test_num * ford_index:]
        pass
    else:
        X_train = np.concatenate((X_[:test_num*ford_index],X_[test_num*(ford_index+1):]))
        Y_train = np.concatenate((Y_[:test_num*ford_index],Y_[test_num*(ford_index+1):]))
        X_test = X_[test_num * ford_index:test_num * (ford_index + 1)]
        Y_test = Y_[test_num * ford_index:test_num * (ford_index + 1)]
    return X_train, Y_train, X_test, Y_test

def normalize(X_):
    return (X_-X_.min(0))/X_.max(axis=0)


if __name__ == '__main__':
    test_something()
    assert False

    loader = DataLoader(2)
    dataloader(2, option_num=0)
    # dataloader(3, option_num=0)
    assert False

    data = loader.data_read()
    # loader.count_col_data(data, 'AD', 5)
    class_name = loader.get_class_name(data, 6)
    # class_name = loader.get_class_name(data, 5)
    for c in class_name:
        loader.count_col_data(data, c, 6)
    # assert False
    loader.extr_data(data, 2)
    # assert False
    # data_tr, label_tr = loader.get_tr_set()
    # data_tst, label_tst = loader.get_tst_set()
    # prt.p_list(data_tr)
    # prt.p_list(label_tr)
