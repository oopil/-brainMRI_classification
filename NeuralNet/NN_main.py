import argparse
from NeuralNet.NN_model import NeuralNet
from NeuralNet.CNN_model import ConvNeuralNet
from NeuralNet.NN_utils import *
'''
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification')
sys.path.append is needed only when using jupyter notebook
'''
'''
when using the all 3 options to the features, 
I could observe high training speed and high testing accuracy.
'''
def parse_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--excel_path', default='/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx', type=str)
    parser.add_argument('--base_folder_path', default='/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0', type=str)
    parser.add_argument('--result_file_name', default='/home/sp/PycharmProjects/brainMRI_classification/nn_result/chosun_MRI_excel_AD_nn_result', type=str)

    parser.add_argument('--neural_net', default='simple', type=str)
    # simple basic
    parser.add_argument('--diag_type', default='clinic', type=str)
    # diag_type = "PET"
    # diag_type = "new"
    # diag_type = "clinic"
    parser.add_argument('--class_option', default='CN vs AD', type=str)
    #PET    # class_option = 'PET pos vs neg'
    #new    # class_option = 'NC vs ADD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    #clinic # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    parser.add_argument('--noise_augment', default=0, type=int)
    parser.add_argument('--class_option_index', default=0, type=int)
    parser.add_argument('--test_num', default=20, type=int)
    parser.add_argument('--fold_num', default=5, type=int)
    parser.add_argument('--excel_option', default='merge', type=str)
    parser.add_argument('--loss_function', default='normal', type=str) # normal / cross_entropy
    # 'None RANDOM SMOTE SMOTEENN SMOTETomek BolderlineSMOTE'
    parser.add_argument('--sampling_option', default='None', type=str)

    parser.add_argument('--is_split_by_num', default=False, type=bool)
    parser.add_argument('--investigate_validation', default=False, type=bool)

    # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
    # BO result -1.2254855784556566, -1.142561108840614
    parser.add_argument('--lr', default=0.01, type=float) #0.01 #0.0602
    parser.add_argument('--weight_stddev', default=0.05, type=float) #0.05 #0.0721
    parser.add_argument('--epoch', default=1500, type=int)
    parser.add_argument('--iter', default=1, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=200, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    # parser.add_argument('--batch_size', default=200, type=int)

    parser.add_argument('--result_dir', default='nn_result', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)

    return parser.parse_args()

def run():
    # parse arguments
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args is None:
        exit()
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        CNN_simple_train(sess, args)
        # NN_cross_validation(sess, args)
        # NN_BayesOptimize(sess, args)

def CNN_simple_train(sess, args):
    CNN = ConvNeuralNet(sess, args)
    CNN.read_cnn_data()
    CNN.build_model()
    # show network architecture
    show_all_variables()
    # launch the graph in a session
    CNN.train()
    # NN.visualize_results(args.epoch - 1)
    print(" [*] Training finished!")

def NN_simple_train(sess, args):
    NN = NeuralNet(sess, args)
    NN.read_nn_data()
    NN.build_model()
    # show network architecture
    show_all_variables()
    # launch the graph in a session
    NN.train()
    # NN.visualize_results(args.epoch - 1)
    print(" [*] Training finished!")

def NN_cross_validation(sess, args):
    NN = NeuralNet(sess, args)
    NN.read_nn_data()
    NN.build_model()
    show_all_variables()

    lr = 10 ** -1.7965511862094083
    w_stddev = 10 ** -1.1072880677553867
    NN.set_lr(lr)
    NN.set_weight_stddev(w_stddev)

    NN.try_all_fold()
    print(" [*] k-fold cross validation finished!")

def NN_BayesOptimize(sess, args):
    NN = NeuralNet(sess, args)
    # target': 87.0, 'params':
    # {'init_learning_rate_log': -1.4511864960726752,
    # 'weight_stddev_log': -1.2848106336275804}}

    #'target': 93.0, 'params':
    # {'init_lr_log': -1.4511864960726752,
    # 'w_stddev_log': -1.2848106336275804}}

    # best score? in NC vs ADD
    # -1.3681144349771235, -1.601517024863694

    # NC vs MCI vs AD
    # -1.7965511862094083, -1.1072880677553867}}
    NN.read_nn_data()
    NN.build_model()
    NN.BayesOptimize()
    print(" [*] Bayesian Optimization finished!")

def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    # NN_cross_validation()
    run()
    # run()
