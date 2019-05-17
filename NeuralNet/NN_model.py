import os
import sys
import subprocess
import tensorflow as tf
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification')
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification/NeuralNet')
# from NeuralNet.neuralnet_ops import *
import NeuralNet.NN_validation as _validation
import NeuralNet.NN_BO as _BO
from NeuralNet.NN_net import *
# import NeuralNet.NN_net as Net
from NeuralNet.NN_ops import *
# import NN_validation as _validation
# import NN_BO as _BO
# from NN_ops import *
from data_merge import *
from bayes_opt import BayesianOptimization

class NeuralNet(object):
    def __init__(self, sess, args):
        self.model_name = "NeuralNet"  # name for checkpoint
        self.sess = sess
        self.excel_path = args.excel_path
        self.base_folder_path = args.base_folder_path
        self.result_file_name = args.result_file_name

        if args.neural_net == 'simple':
            self.model_name = str('simple.... change it later')
        if args.neural_net == 'basic':
            self.model_name = self.neural_net_basic
        if args.neural_net == 'attention':
            self.model_name = self.neural_net_attention
        if args.neural_net == 'attention_self':
            self.model_name = self.neural_net_self_attention
        if args.neural_net == 'attention_often':
            self.model_name = self.neural_net_attention_often

        self.diag_type = args.diag_type
        self.excel_option = args.excel_option
        self.test_num = args.test_num
        self.fold_num = args.fold_num
        self.is_split_by_num = args.is_split_by_num
        self.sampling_option = args.sampling_option
        self.learning_rate = args.lr
        self.loss_function = args.loss_function
        self.investigate_validation = args.investigate_validation
        self.weight_stddev = args.weight_stddev
        self.weight_initializer = tf.random_normal_initializer(mean=0., stddev=self.weight_stddev)

        self.class_option = args.class_option
        self.class_option_index = args.class_option_index
        class_split = self.class_option.split('vs')
        self.class_num = len(class_split)
        self.noise_augment_stddev = args.noise_augment

        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.epoch = args.epoch
        self.iteration = args.iter
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.summary_freq = args.summary_freq

        self.result_file_name = self.result_file_name + self.diag_type +'_' +self.class_option
        # self.iteration = args.iteration
        # self.batch_size = args.batch_size
        self.is_print = True

        self.args = args

    def print_arg(self, args):
        print()
        print("##### Information #####")
        for i, arg in enumerate(vars(args)):
            print(i, arg, getattr(args, arg))

    ##################################################################################
    # Set private variable
    ##################################################################################
    def set_weight_stddev(self, stddev):
        self.weight_stddev = stddev
        self.weight_initializer = tf.random_normal_initializer(mean=0., stddev=self.weight_stddev)
        print('weight standard deviance is set to : {}' .format(self.weight_stddev))

    def set_lr(self, lr):
        self.learning_rate = lr
        print('learning rate is set to : {}' .format(self.learning_rate))

    def set_model(self, model):
        pass
    ##################################################################################
    # Dataset
    ##################################################################################
    def read_nn_data(self):
        # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
        sampling_option_str = 'None RANDOM SMOTE SMOTEENN SMOTETomek BolderlineSMOTE'  # ADASYN
        sampling_option_split = sampling_option_str.split(' ')
        whole_set = NN_dataloader(self.diag_type, self.class_option,\
                                  self.excel_path, self.excel_option, self.test_num, self.fold_num, self.is_split_by_num)
        whole_set = np.array(whole_set)
        self.train_data, self.train_label, self.test_data, self.test_label = whole_set[0]
        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)
        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)
        self.test_data, self.test_label = valence_class(self.test_data, self.test_label, self.class_num)
        self.train_data, self.train_label = over_sampling(self.train_data, self.train_label, self.sampling_option)
        self.input_feature_num = len(self.train_data[0])

    def noise_addition(self, data):
        return gaussian_noise_layer(data, std=self.noise_augment_stddev)

    ##################################################################################
    # validation
    ##################################################################################
    def try_all_fold(self):
        result_list = _validation.try_all_fold(self)
        _validation.save_results(self, result_list)

    # def BO_train_and_validate(self, init_lr_log, w_stddev_log):
    def BO_train_and_validate(self, init_lr_log, w_stddev_log):
        self.set_lr(10**init_lr_log)
        self.set_weight_stddev(10**w_stddev_log)
        print('-'*100)
        # print('learning rate : {}\nstddev of weight : {}'.\
        #       format(self.learning_rate, 10**w_stddev_log))
        print('learning rate : {}\nstddev of weight : {}'.\
              format(self.learning_rate, 10**w_stddev_log))
        return self.train()

    def BayesOptimize(self, init_lr_log, w_stddev_log):
        _BO.BayesOptimize(init_lr_log, w_stddev_log)

    # def BayesOptimize(self):
    #     bayes_optimizer = BayesianOptimization(
    #         f=self.BO_train_and_validate,
    #         pbounds={
    #             # 78 <= -1.2892029132535314,-1.2185073691640054
    #             # 85 <= -1.2254855784556566, -1.142561108840614}}
    #             'init_lr_log': (-2.0,-1.0),
    #             'w_stddev_log': (-2.0,-1.0)
    #         },
    #         random_state=0,
    #         # verbose=2
    #     )
    #     bayes_optimizer.maximize(
    #         init_points=5,
    #         n_iter=40,
    #         acq='ei',
    #         xi=0.01
    #     )
    #     BO_results = []
    #     BO_results.append('\n\t\t<<< class option : {} >>>\n' .format(self.class_option))
    #     BO_result_file_name = "BO_result/BayesOpt_results"\
    #                           + str(time.time()) + '_' + self.class_option
    #     fd = open(BO_result_file_name, 'a+t')
    #     for i, ressult in enumerate(bayes_optimizer.res):
    #         BO_results.append('Iteration {}:{}\n'.format(i, ressult))
    #         print('Iteration {}: {}'.format(i, ressult))
    #         fd.writelines('Iteration {}:{}\n'.format(i, ressult))
    #     BO_results.append('Final result: {}\n'.format(bayes_optimizer.max))
    #     fd.writelines('Final result: {}\n'.format(bayes_optimizer.max))
    #     print('Final result: {}\n'.format(bayes_optimizer.max))
    #     fd.close()
    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        """ Graph Input """
        self.input = tf.placeholder(tf.float32, [None, self.input_feature_num], name='inputs')
        # self.label = tf.placeholder(tf.float32, [None, self.class_num], name='targets')
        self.label = tf.placeholder(tf.int32, [None], name='targets')
        self.label_onehot = onehot(self.label, self.class_num)
        print(self.input)

        # self.logits = self.model_name(self.input, reuse=False)
        self.my_model = SimpleNet(tf.truncated_normal_initializer, tf.nn.relu, self.class_num)
        # self.my_model = ResNet(tf.truncated_normal_initializer, tf.nn.relu, self.class_num)
        # self.logits = Net.NeuralNetSimple(self.input, tf.truncated_normal_initializer, tf.nn.relu, self.class_num)
        self.logits = self.my_model.model(self.input)
        self.pred = tf.argmax(self.logits,1)
        self.accur = accuracy(self.logits, self.label_onehot) // 1

        # get loss for discriminator
        """ Loss Function """
        with tf.name_scope('Loss'):
            # self.loss = classifier_loss('normal', predictions=self.logits, targets=self.label_onehot)
            self.loss = classifier_loss(self.loss_function, predictions=self.logits, targets=self.label_onehot)
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if 'neuralnet' in var.name]

        # optimizers
        start_lr = self.learning_rate
        global_step = tf.Variable(0, trainable=False)
        total_learning = self.epoch
        lr = tf.train.exponential_decay(start_lr, global_step, decay_steps=self.epoch//100, decay_rate=.96, staircase=True)

        self.optim = tf.train.AdamOptimizer(lr).minimize(self.loss)
        # self.d_optim = tf.train.AdamOptimizer(d_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.loss, var_list=d_vars)
        #self.d_optim = tf.train.AdagradOptimizer(d_lr).minimize(self.loss, var_list=d_vars)

        """ Summary """
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar('accuracy', self.accur)
        self.merged_summary = tf.summary.merge_all()

    ##################################################################################
    # Train
    ##################################################################################
    def train(self):
        #--------------------------------------------------------------------------------------------------
        # initialize all variables
        tf.global_variables_initializer().run()
        # graph inputs for visualize training results
        # saver to save model
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir +'_train', self.sess.graph)
        self.train_writer.add_graph(self.sess.graph)

        if self.investigate_validation:
            self.test_writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir +'_test', self.sess.graph)
            self.test_writer.add_graph(self.sess.graph)
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        # loop for epoch
        start_time = time.time()
        past_loss = -1.

        self.valid_accur = []
        self.train_accur = []
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.iteration):
                #---------------------------------------------------
                if self.noise_augment_stddev:
                    train_data = self.noise_addition(self.train_data)
                train_feed_dict = {
                    self.input: train_data,
                    self.label: self.train_label
                }
                _, merged_summary_str, loss, pred, accur = self.sess.run(\
                    [self.optim, self.merged_summary, self.loss, self.pred, self.accur], \
                    feed_dict=train_feed_dict)
                # self.train_writer.add_summary(merged_summary_str, global_step=counter)
                if epoch % self.print_freq == 0:
                        print("Epoch: [{}/{}] [{}/{}], loss: {}, accur: {}"\
                              .format(epoch, self.epoch, idx, self.iteration,loss, accur))
                        # print("Epoch: [%2d/%2d] [%5d/%5d] time: %4.4f, loss: %.8f" \
                        #       % (epoch, self.epoch, idx, self.iteration, time.time() - start_time, loss))
                        # print("pred : {}".format(self.train_label))
                        # print("pred : {}".format(pred))
                        test_accur, test_summary = self.test(counter)
                        self.valid_accur.append(test_accur)
                        self.train_accur.append(accur)
                        print('=' * 100)

                if epoch % self.summary_freq == 0:
                    self.train_writer.add_summary(merged_summary_str, global_step=counter)
                    if self.investigate_validation:
                        self.test(counter)
            counter+=1

        print(self.train_accur)
        print(self.valid_accur)
        return np.max(self.valid_accur)

    def test(self, counter):
        test_feed_dict = {
            self.input: self.test_data,
            self.label: self.test_label
        }
        # tf.global_variables_initializer().run()
        loss, accur, pred, merged_summary_str = self.sess.run([self.loss, self.accur, self.pred, self.merged_summary], feed_dict=test_feed_dict)

        # self.test_writer.add_summary(merged_summary_str, counter)
        if self.investigate_validation:
            pass
        else:
            print("Test result => accur : {}, loss : {}".format(accur, loss))
            print("pred : {}".format(self.test_label))
            print("pred : {}".format(pred))
        return accur, merged_summary_str

    def simple_test(self):
        test_feed_dict = {
            self.input: self.test_data,
            self.label: self.test_label
        }
        loss, accur, pred = self.sess.run([self.loss, self.accur, self.pred], feed_dict=test_feed_dict)
        return accur

    def simple_train(self):
        tf.global_variables_initializer().run()
        start_epoch = 0
        start_batch_id = 0
        self.valid_accur = []
        self.train_accur = []
        for epoch in range(start_epoch, self.epoch):
            for idx in range(start_batch_id, self.iteration):
                # ---------------------------------------------------
                train_data = self.noise_addition(self.train_data)
                train_feed_dict = {
                    self.input: train_data,
                    self.label: self.train_label
                }
                _, loss, pred, accur = self.sess.run( \
                    [self.optim, self.loss, self.pred, self.accur], \
                    feed_dict=train_feed_dict)
                if epoch % self.print_freq == 0:
                    self.valid_accur.append(self.simple_test())
                    self.train_accur.append(accur)
        return self.valid_accur, self.train_accur

    @property
    def model_dir(self):
        return "{}".format(self.model_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, str(self.model_name)+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save_result(self, contents):
        result_file_name = \
            '/home/sp/PycharmProjects/brainMRI_classification/regression_result/chosun_MRI_excel_logistic_regression_result_' \
            + diag_type + '_' + class_option
        is_remove_result_file = True
        if is_remove_result_file:
            # command = 'rm {}'.format(result_file_name)
            # print(command)
            subprocess.call(['rm', result_file_name])
            # os.system(command)
        # assert False
        line_length = 100
        pass