import time
import sys
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification')
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification/NeuralNet')
# from NeuralNet.neuralnet_ops import *
from NeuralNet.NN_ops import *
from data_merge import *

def try_all_fold(self)->list:
    now = time.gmtime(time.time())
    time_date = [now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min]
    whole_set = NN_dataloader(self.diag_type, self.class_option, \
                              self.excel_path, self.excel_option, self.test_num, self.fold_num,
                              self.is_split_by_num)
    whole_set = np.array(whole_set)
    result_list = []
    top_train_accur_list = []
    top_valid_accur_list = []

    ##########################################################################
    # No sampling
    ##########################################################################
    # self.sampling_option = 'None'
    # result_list.append('\n\t\t<<< class option : {} / oversample : {} >>>\n' \
    #                    .format(self.class_option, self.sampling_option))
    # result_list.append('model : {}, lr : {}, epoch : {}, noise augment : {},  {} \n' \
    #                    .format(self.args.neural_net, self.learning_rate, \
    #                            self.epoch, self.noise_augment, now.tm_year))
    # for i, fold in enumerate(whole_set):
    #     self.train_data, self.train_label, self.test_data, self.test_label = fold
    #     self.test_data, self.test_label = valence_class(self.test_data, self.test_label, self.class_num)
    #     self.train_data, self.train_label = over_sampling(self.train_data, self.train_label, self.sampling_option)
    #     valid_result, train_result = self.simple_train()
    #     top_valid_accur = np.max(valid_result, 0)
    #     top_train_accur = np.max(train_result, 0)
    #     result_list.append('\n[ fold : {}/{:<3} ] train/test : {}/{} \ntop train : {}\ntop test : {}\n' \
    #                        .format(i, self.fold_num, len(self.train_label), len(self.test_label), top_train_accur,
    #                                top_valid_accur))
    #     result_list.append('train {}\n'.format([train_result]))
    #     result_list.append('test  {}\n'.format([valid_result]))
    #     top_train_accur_list.append(top_train_accur)
    #     top_valid_accur_list.append(top_valid_accur)
    # result_list.append('[[ avg top train : {}, avg top test : {} ]]\n{}\n{}' \
    #                    .format(np.mean(top_train_accur_list), np.mean(top_valid_accur_list), top_train_accur_list,
    #                            top_valid_accur_list))

    ##########################################################################
    # Random sampling
    ##########################################################################
    top_train_accur_list = []
    top_valid_accur_list = []
    train_saturation_list = []
    valid_saturation_list = []

    result_list.append('=' * 100)
    self.sampling_option = 'RANDOM'
    result_list.append('\n\t\t<<< class option : {} / oversample : {} >>>\n' \
                       .format(self.class_option, self.sampling_option))
    result_list.append('model : {}, lr : {}, epoch : {}, noise augment : {},  {} \n' \
                       .format(self.args.neural_net, self.learning_rate, \
                               self.epoch, self.noise_augment_stddev, now.tm_year))
    for i, fold in enumerate(whole_set):
        self.train_data, self.train_label, self.test_data, self.test_label = fold
        self.test_data, self.test_label = valence_class(self.test_data, self.test_label, self.class_num)
        self.train_data, self.train_label = over_sampling(self.train_data, self.train_label, self.sampling_option)
        valid_result, train_result = self.simple_train()
        top_valid_accur = np.max(valid_result, 0)
        top_train_accur = np.max(train_result, 0)
        result_list.append('\n[ fold : {}/{:<3} ]\ntop train : {}\ntop test : {}\n' \
                           .format(i, self.fold_num, top_train_accur, top_valid_accur))
        result_list.append('train {}\n'.format([train_result]))
        result_list.append('test  {}\n'.format([valid_result]))
        top_train_accur_list.append(top_train_accur)
        top_valid_accur_list.append(top_valid_accur)

        saturation_count = 5
        train_saturation_list.append(np.mean(train_result[-saturation_count:]))
        valid_saturation_list.append(np.mean(valid_result[-saturation_count:]))

    print(top_train_accur_list, np.mean(top_train_accur_list))
    print(top_valid_accur_list, np.mean(top_valid_accur_list))
    # assert False
    result_list.append('[[ avg top train : {}, avg top test : {} ]]\n{}\n{}\n' .format(np.mean(top_train_accur_list), np.mean(top_valid_accur_list), top_train_accur_list, top_valid_accur_list))
    result_list.append('[[ avg saturaion train : {}, avg saturation test : {} ]]\n{}\n{}\n' .format(np.mean(train_saturation_list), np.mean(valid_saturation_list), train_saturation_list, valid_saturation_list))

    for result in result_list:
        print(result)
    return result_list

def save_results(self, result_list)->None:
    is_remove_result_file = False
    # is_remove_result_file = True
    if is_remove_result_file:
        subprocess.call(['rm', self.result_file_name])
        # os.system(command)
    file = open(self.result_file_name, 'a+t')
    # print('<< results >>')
    for result in result_list:
        file.writelines(result)