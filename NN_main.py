import argparse
from NeuralNet.NN_model import NeuralNet
from NeuralNet.NN_utils import *
'''
sys.path.append('/home/soopil/PycharmProjects/brainMRI_classification')
sys.path.append is needed only when using jupyter notebook
'''
'''
when using the all 3 options to the features, 
I could observe high training speed and high testing accuracy.
'''
def parse_args() -> argparse:
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',                default='186', type=str)
    parser.add_argument('--task',               default='cv', type=str) # train cv bo
    parser.add_argument('--setting',            default='desktop', type=str)
    parser.add_argument('--result_file_name',   default='/home/soopil/Desktop/github/brainMRI_classification/nn_result/chosun_MRI_excel_AD_nn_result', type=str)
    # 'None RANDOM SMOTE SMOTEENN SMOTETomek BolderlineSMOTE'
    parser.add_argument('--excel_option',       default='merge', type=str)
    parser.add_argument('--is_split_by_num',    default=False, type=str2bool)
    parser.add_argument('--investigate_validation', default=False, type=str2bool)
    parser.add_argument('--iter',               default=1, type=int)
    parser.add_argument('--summary_freq',       default=100, type=int)
    parser.add_argument('--class_option_index', default=0, type=int)
    parser.add_argument('--test_num',           default=20, type=int)
    parser.add_argument('--fold_num',           default=5, type=int)
    parser.add_argument('--result_dir',         default='nn_result', type=str)
    parser.add_argument('--log_dir',            default='log', type=str)
    parser.add_argument('--checkpoint_dir',     default='checkpoint', type=str)

    parser.add_argument('--epoch',              default=200, type=int)
    parser.add_argument('--print_freq',         default=5, type=int)
    parser.add_argument('--save_freq',          default=200, type=int)
    # diag_type = PET new clinic
    '''
        from this line, i need to save information after running.
        start with 19 index.
    '''
    # neural_net ## simple basic attention self_attention attention_often
    # conv_neural_net ## simple, basic attention
    parser.add_argument('--neural_net',         default='simple', type=str)
    parser.add_argument('--class_option',       default='clinic CN vs AD', type=str)
    #PET    # class_option = 'PET pos vs neg'
    #new    # class_option = 'NC vs ADD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    #clinic # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    parser.add_argument('--lr',                 default=0.0001, type=float) #0.001 #0.0602
    parser.add_argument('--patch_size',         default=48, type=int)
    parser.add_argument('--batch_size',         default=1, type=int)
    parser.add_argument('--weight_stddev',      default=0.05, type=float) #0.05 #0.0721
    parser.add_argument('--loss_function',      default='cEntropy', type=str) # L2 / cross
    parser.add_argument('--sampling_option',    default='RANDOM', type=str)
    parser.add_argument('--noise_augment',      default=0.1, type=float)
    # if i use this nosie augment, the desktop stop
    # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
    # BO result -1.2254855784556566, -1.142561108840614
    parser.add_argument('--excel_path',         default='None', type=str)
    parser.add_argument('--base_folder_path',   default='None', type=str)
    parser.add_argument('--diag_type',          default='None', type=str)
    return parser.parse_args()

def args_set(args):
    sv_set_dict = {
        "desktop": 0,
        "sv186": 186,
        "sv144": 144,
        "sv202": 202,
    }
    sv_set = sv_set_dict[args.setting]
    if sv_set == 186:
        # args.base_folder_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_empty_copy'
        args.excel_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 0:  # desktop
        # args.base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        args.excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    # elif sv_set == 144:  # desktop
    #     args.base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
    #     args.excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 202:  # desktop
        # args.base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        args.excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'

    class_option = args.class_option.split(' ')
    args.diag_type = class_option[0]
    args.class_option = ' '.join(class_option[1:])
    return args

def run():
    # parse arguments
    args = parse_args()
    args = args_set(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args is None:
        exit()
    # open session
    task = None
    if args.task == 'cv':
        task = NN_cross_validation
    elif args.task == 'train':
        task = NN_simple_train
    elif args.task == 'bo':
        task = NN_BayesOptimize
    assert task != None

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        task(sess, args)

def NN_simple_train(sess, args):
    NN = NeuralNet(sess, args)
    NN.print_arg(args)
    NN.read_nn_data()
    # assert False
    # show network architecture
    show_all_variables()
    # i think we should set this param before build model
    # NN.set_lr(10 ** -1.7965511862094083)
    # NN.set_weight_stddev(10 ** -1.1072880677553867)
    NN.build_model()
    # launch the graph in a session
    NN.train()
    # NN.visualize_results(args.epoch - 1)
    print(" [*] Training finished!")

def NN_cross_validation(sess, args):
    NN = NeuralNet(sess, args)
    NN.print_arg(args)
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
    NN.print_arg(args)
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

