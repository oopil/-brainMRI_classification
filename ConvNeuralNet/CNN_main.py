import argparse
from ConvNeuralNet.CNN_model import ConvNeuralNet
from ConvNeuralNet.CNN_utils import *
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
    parser.add_argument('--gpu',                default='0', type=str)
    parser.add_argument('--task',               default='train', type=str) # train cv bo
    parser.add_argument('--setting',            default='desktop', type=str)
    parser.add_argument('--mask',               default=False, type=str2bool)
    parser.add_argument('--buffer_scale',       default=10, type=int)

    parser.add_argument('--investigate_validation', default=False, type=str2bool)
    parser.add_argument('--result_dir',         default='nn_result', type=str)
    parser.add_argument('--log_dir',            default='log', type=str)
    parser.add_argument('--checkpoint_dir',     default='checkpoint', type=str)
    parser.add_argument('--iter',               default=1, type=int)
    parser.add_argument('--class_option_index', default=0, type=int)
    parser.add_argument('--fold_num',           default=5, type=int)

    parser.add_argument('--print_freq',         default=1, type=int)
    parser.add_argument('--summary_freq',       default=100, type=int)
    parser.add_argument('--save_freq',          default=200, type=int)
    parser.add_argument('--epoch',              default=100, type=int)

    parser.add_argument('--network',    default='siam', type=str) # simple attention siam
    parser.add_argument('--class_option',       default='clinic CN vs AD', type=str)
    # PET    # class_option = 'PET pos vs neg'
    # new    # class_option = 'NC vs ADD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    # clinic # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    parser.add_argument('--lr',                 default=0.00001, type=float)  # 0.001 #0.0602
    parser.add_argument('--patch_size',         default=48, type=int)
    parser.add_argument('--batch_size',         default=30, type=int)
    parser.add_argument('--weight_stddev',      default=0.05, type=float)  # 0.05 #0.0721
    parser.add_argument('--loss_function',      default='cEntropy', type=str)  # L2 / cross
    parser.add_argument('--sampling_option',    default='RANDOM', type=str)
    parser.add_argument('--noise_augment',      default=0.1, type=float)

    parser.add_argument('--result_file_name',default='/home/soopil/Desktop/github/brainMRI_classification/nn_result/chosun_MRI_excel_AD_nn_result', type=str)
    parser.add_argument('--excel_path',         default='None', type=str)
    parser.add_argument('--base_folder_path',   default='None', type=str)
    parser.add_argument('--diag_type',          default='None', type=str)
    parser.add_argument('--excel_option',       default='P', type=str) # P V T merge
    return parser.parse_args()

def args_set(args):
    sv_set_dict = {
        "desktop": 0,
        "sv186": 186,
        "sv144": 144,
        "sv202": 202,
    }
    sv_set = sv_set_dict[args.setting]
    if sv_set == 186: # server186
        args.base_folder_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_empty_copy'
        args.excel_path = '/home/public/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 0:  # desktop
        args.base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        args.excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 144:  # server144
        args.base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
        args.excel_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_test.xlsx'
    elif sv_set == 202:  # server202
        args.base_folder_path = '/home/soopil/Desktop/Dataset/MRI_chosun/ADAI_MRI_Result_V1_0_processed'
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
    task_func = None
    if args.task == 'cv':
        task_func = CNN_simple_train
    elif args.task == 'train':
        task_func = CNN_simple_train
    elif args.task == 'bo':
        task_func = CNN_simple_train
    assert task_func != None

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        task_func(sess, args)

def CNN_simple_train(sess, args):
    CNN = ConvNeuralNet(sess, args)
    CNN.read_cnn_data()
    # show network architecture
    # launch the graph in a session
    # CNN.test_data_read() # test only data reading
    # CNN.set_lr(.001)
    # CNN.set_weight_stddev(.05)
    CNN.build_model()
    show_all_variables()
    CNN.train()
    # NN.visualize_results(args.epoch - 1)
    print(" [*] Training finished!")

def print_result_file(result_file_name):
    file = open(result_file_name, 'rt')
    lines = file.readlines()
    for line in lines:
        print(line)
    file.close()

if __name__ == '__main__':
    run()

