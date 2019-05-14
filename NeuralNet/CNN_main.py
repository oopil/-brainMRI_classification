import argparse
from NeuralNet.NN_model import NeuralNet
from NeuralNet.CNN_model import ConvNeuralNet
from NeuralNet.NN_utils import *
'''
sys.path.append('/home/soopil/PycharmProjects/brainMRI_classification')
sys.path.append is needed only when using jupyter notebook
'''
'''
when using the all 3 options to the features, 
I could observe high training speed and high testing accuracy.
'''
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args() -> argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',                default='0', type=str)
    parser.add_argument('--setting',            default='desktop', type=str)
    parser.add_argument('--mask',               default=True, type=str2bool)
    parser.add_argument('--buffer_scale',       default=3, type=int)

    parser.add_argument('--result_file_name',default='/home/soopil/Desktop/github/brainMRI_classification/nn_result/chosun_MRI_excel_AD_nn_result', type=str)
    parser.add_argument('--is_split_by_num',    default=False, type=str2bool)
    parser.add_argument('--investigate_validation', default=False, type=str2bool)
    parser.add_argument('--result_dir',         default='nn_result', type=str)
    parser.add_argument('--log_dir',            default='log', type=str)
    parser.add_argument('--checkpoint_dir',     default='checkpoint', type=str)
    parser.add_argument('--iter',               default=1, type=int)
    parser.add_argument('--class_option_index', default=0, type=int)
    parser.add_argument('--test_num',           default=20, type=int)
    parser.add_argument('--fold_num',           default=5, type=int)

    parser.add_argument('--print_freq',         default=1, type=int)
    parser.add_argument('--summary_freq',       default=100, type=int)
    parser.add_argument('--save_freq',          default=200, type=int)

    parser.add_argument('--diag_type',          default='clinic', type=str)
    # diag_type = "PET"
    # diag_type = "new"
    # diag_type = "clinic"
    '''
        from this line, i need to save information after running.
        start with 19 index.
    '''
    # conv_neural_net ## simple, basic attention
    parser.add_argument('--conv_neural_net',    default='simple', type=str)
    parser.add_argument('--class_option',       default='CN vs AD', type=str)
    #PET    # class_option = 'PET pos vs neg'
    #new    # class_option = 'NC vs ADD'  # 'aAD vs ADD'#'NC vs ADD'#'NC vs mAD vs aAD vs ADD'
    #clinic # class_option = 'MCI vs AD'#'MCI vs AD'#'CN vs MCI'#'CN vs AD' #'CN vs MCI vs AD'
    parser.add_argument('--lr',                 default=0.001, type=float) #0.01 #0.0602
    parser.add_argument('--patch_size',         default=48, type=int)
    parser.add_argument('--batch_size',         default=5, type=int)
    parser.add_argument('--weight_stddev',      default=0.05, type=float)
    #0.05 #0.0721
    parser.add_argument('--epoch',              default=10, type=int)
    parser.add_argument('--loss_function',      default='normal', type=str)
    # normal / cross_entropy
    parser.add_argument('--sampling_option',    default='RANDOM', type=str)
    parser.add_argument('--noise_augment',      default=True, type=str2bool)
    # BO result -1.2254855784556566, -1.142561108840614
    return parser.parse_args()

def run():
    # parse arguments
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sv_set_dict = {
        "desktop": 0,
        "sv186": 186,
        "sv144": 144,
        "sv202": 202,
    }
    sv_set = sv_set_dict[args.setting]
    if args is None:
        exit()
    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        CNN_simple_train(sess, args)

def CNN_simple_train(sess, args):
    CNN = ConvNeuralNet(sess, args)
    CNN.read_cnn_data()
    # show network architecture
    # launch the graph in a session
    # CNN.test_data_read() # test only data reading
    CNN.set_lr(.001)
    CNN.set_weight_stddev(.05)
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

