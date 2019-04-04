from data_merge import *
from sklearn.linear_model import LogisticRegression
from data import *

'''
    set the data option and load dataset
'''
base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
# base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'
excel_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
# "clinic" or "new" or "PET"
# 'PET pos vs neg', 'NC vs MCI vs AD' 'NC vs mAD vs aAD vs ADD'
diag_type = "PET"
class_option = 'PET pos vs neg'
# diag_type = "new"
# class_option = 'NC vs mAD vs aAD vs ADD'
# diag_type = "clinic"
# class_option = 'NC vs AD' #'NC vs MCI vs AD'
excel_option = 'merge' # P V T merge
test_num = 20
fold_num = 5
is_split_by_num = False
whole_set = NN_dataloader(diag_type, class_option, base_folder_path, \
                  excel_path, excel_option, test_num, fold_num, is_split_by_num)
print(len(whole_set))


def logistic_regression(one_dataset):
    train_data, train_label, test_data, test_label = one_dataset
    print(test_label)
    assert False

    test_data, test_label = valence_class(test_data, test_label, class_num)
    # train_data, train_label = valence_class(train_data, train_label, class_num)
    print(train_data.shape, test_data.shape)

    if class_num == 2:
        logreg = LogisticRegression(solver='lbfgs')
    elif class_num == 3:
        logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial') # multinomial / auto/ ovr
    logreg.max_iter = 1000
    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(train_data, train_label)

    Pred = logreg.predict(test_data)
    print('label\t:',test_label)
    print('predict :',Pred)
    total_num = len(test_label)
    correct_answer = 0
    for i in range(total_num):
        if test_label[i] == Pred[i]:
            correct_answer += 1

    print('the probability is ')
    prob = correct_answer*100 / total_num
    print(prob)

def main():
    for one_fold_set in whole_set:
        logistic_regression(one_fold_set)
        pass

if __name__ == '__main__':
    main()