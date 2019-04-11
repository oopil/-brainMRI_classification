import argparse
from NeuralNet.neuralnet_model import NeuralNet
from NeuralNet.neuralnet_utils import *
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
    parser.add_argument('--diag_type', default='new', type=str)
    # diag_type = "PET"
    # diag_type = "new"
    # diag_type = "clinic"
    parser.add_argument('--class_option', default='NC vs aAD', type=str)
    #PET    # class_option = 'PET pos vs neg'
    #new    # class_option = 'NC vs ADD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    #clinic # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    parser.add_argument('--class_option_index', default=0, type=int)
    parser.add_argument('--excel_option', default='merge', type=str)
    parser.add_argument('--loss_function', default='normal', type=str) # normal / cross_entropy
    parser.add_argument('--test_num', default=20, type=int)
    parser.add_argument('--fold_num', default=5, type=int)
    parser.add_argument('--is_split_by_num', default=False, type=bool)
    parser.add_argument('--sampling_option', default='RANDOM', type=str)
    # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
    parser.add_argument('--lr', default=0.01, type=float) #0.01
    parser.add_argument('--epoch', default=4000, type=int)
    parser.add_argument('--iter', default=1, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=200, type=int)
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
        # graph = tf.Graph()
        # with graph.as_default(): # ?????
        NN = NeuralNet(sess, args)
        whole_set = NN.read_nn_data()
        # assert False
        # build graph
        NN.build_model()
        # show network architecture
        show_all_variables()
        # launch the graph in a session
        # NN.try_all_fold()
        NN.train()

        # assert False
        # visualize learned generator
        # NN.visualize_results(args.epoch - 1)
        print(" [*] Training finished!")
        # best_score = NN.test()
        # print(" [*] Test finished!")
        # return best_score

# In[40]:
''''''


def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    run()

