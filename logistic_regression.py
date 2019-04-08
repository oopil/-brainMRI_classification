import os
import argparse
import datetime
import subprocess
from data_merge import *
from sklearn.linear_model import LogisticRegression

def logistic_regression(one_dataset, sampling_option, class_num)->str and int:
    train_data, train_label, test_data, test_label = one_dataset
    test_data, test_label = valence_class(test_data, test_label, class_num)
    train_data, train_label = over_sampling(train_data, train_label, sampling_option)
    # train_data, train_label = valence_class(train_data, train_label, class_num)
    # print(train_data.shape, test_data.shape)

    if class_num == 2:
        logreg = LogisticRegression(solver='lbfgs')
    elif class_num > 2:
        logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial') # multinomial / auto/ ovr
    logreg.max_iter = 1000
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(train_data, train_label)

    Pred = logreg.predict(train_data)
    print('label\t:', train_label)
    print('predict :', Pred)
    total_num = len(train_label)
    correct_answer = 0
    for i in range(total_num):
        if train_label[i] == Pred[i]:
            correct_answer += 1

    train_accur = correct_answer * 100 / total_num
    # print('the probability is {}'.format(train_accur))

    Pred = logreg.predict(test_data)
    print('label\t:',test_label)
    print('predict :',Pred)
    total_num = len(test_label)
    correct_answer = 0
    for i in range(total_num):
        if test_label[i] == Pred[i]:
            correct_answer += 1
    # print('the probability is {}'.format(test_accur))

    test_accur = correct_answer*100 / total_num
    print('the probability is {}'.format(test_accur))
    return 'train and test number : {} / {:<5}'.format(len(train_label),len(test_label))+\
           ',top Test  : {:<10}' .format(test_accur // 1)+\
           ',top Train : {:<10}\n'.format(train_accur // 1), test_accur

def main():
    '''
        set the data option and load dataset
    '''
    base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
    excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
    # "clinic" or "new" or "PET"
    # 'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
    # diag_type = "PET"
    # class_option = 'PET pos vs neg'
    diag_type = "new"
    class_option = 'NC vs ADD'#'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    # diag_type = "clinic"
    # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    class_split = class_option.split('vs')
    class_num = len(class_split)
    excel_option = 'merge'  # P V T merge
    test_num = 20
    fold_num = 5
    is_split_by_num = False # split the dataset by fold.
    # sampling_option = 'SMOTENC'
    # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
    sampling_option_str = 'None RANDOM SMOTE SMOTEENN SMOTETomek BolderlineSMOTE'# ADASYN
    sampling_option_split = sampling_option_str.split(' ')

    whole_set = NN_dataloader(diag_type, class_option, \
                              excel_path, excel_option, test_num, fold_num, is_split_by_num)
    whole_set = np.array(whole_set)

    result_file_name = \
    '/home/sp/PycharmProjects/brainMRI_classification/regression_result/chosun_MRI_excel_logistic_regression_result_'\
    +diag_type +'_'+ class_option
    '''
    if there is space in the file name, i can't use it in the linux command.
    '''
    is_remove_result_file = True
    if is_remove_result_file:
        # command = 'rm {}'.format(result_file_name)
        # print(command)
        subprocess.call(['rm',result_file_name])
        # os.system(command)
    # assert False
    line_length = 100

    total_test_accur = []
    for sampling_option in sampling_option_split:
        results = []
        test_accur_list = []
        results.append('\n\t\t<<< class option : {} / oversample : {} >>>\n'.format(class_option, sampling_option))
        date = str(datetime.datetime.now())+'\n'
        results.append(date)
        # assert False
        print(len(whole_set))
        for fold_index, one_fold_set in enumerate(whole_set):
            train_num, test_num = len(one_fold_set[0]), len(one_fold_set[2])
            contents = []
            contents.append('fold : {}/{:<3},'.format(fold_index, fold_num))
            line, test_accur = logistic_regression(one_fold_set, sampling_option, class_num)
            contents.append(line)
            test_accur_list.append(test_accur)
            results.append(contents)

        test_accur_avg = int(sum(test_accur_list)/len(test_accur_list))
        results.append('{} : {}\n'.format('avg test accur',test_accur_avg))
        results.append('=' * line_length + '\n')
        total_test_accur.append(test_accur_avg)

        file = open(result_file_name, 'a+t')
        # print('<< results >>')
        for result in results:
            file.writelines(result)
            # print(result)
        # print(contents)
        file.close()
    print_result_file(result_file_name)
    print(total_test_accur)

def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    main()

    '''
    def parse_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_option_index', type=int, default='0')
    parser.add_argument('--ford_num', type=int, default='5')
    parser.add_argument('--ford_index', type=int, default='0')
    parser.add_argument('--keep_prob', type=float, default='0.9')
    parser.add_argument('--lr', type=float, default='0.01')
    parser.add_argument('--epochs', type=int, default='2000')
    parser.add_argument('--save_freq', type=int, default='300')
    parser.add_argument('--print_freq', type=int, default='100')

    return parser.parse_args()
    '''