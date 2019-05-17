import sys
import tensorflow as tf
sys.path.append('..')
sys.path.append('/home/soopil/Desktop/github/brainMRI_classification')
sys.path.append('/home/soopil/Desktop/github/brainMRI_classification/NeuralNet')
# import ConvNeuralNet.CNN_validation as _validation
# import ConvNeuralNet.CNN_BO as _BO
from ConvNeuralNet.CNN_data import *
from ConvNeuralNet.CNN_net import *
from data_merge import *

class ConvNeuralNet:
    def __init__(self, sess, args):
        self.model_name = args.network  # name for checkpoint
        self.sess = sess
        self.excel_path = args.excel_path
        self.base_folder_path = args.base_folder_path
        self.result_file_name = args.result_file_name

        self.diag_type = args.diag_type
        self.excel_option = args.excel_option
        self.fold_num = args.fold_num
        self.sampling_option = args.sampling_option
        self.learning_rate = args.lr
        self.loss_function = args.loss_function
        self.investigate_validation = args.investigate_validation
        self.weight_stddev = args.weight_stddev
        self.weight_initializer = tf.random_normal_initializer(mean=0., stddev=self.weight_stddev)

        self.class_option = args.class_option
        class_split = self.class_option.split('vs')
        self.class_num = len(class_split)
        self.noise_augment = args.noise_augment

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
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.is_print = True
        self.args = args

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
    ##################################################################################
    # Dataset
    ##################################################################################
    def read_cnn_data(self):
        # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
        sampling_option_str = 'None RANDOM SMOTE SMOTEENN SMOTETomek BolderlineSMOTE'  # ADASYN
        sampling_option_split = sampling_option_str.split(' ')
        whole_set = CNN_dataloader(self.base_folder_path, self.diag_type, self.class_option, self.excel_path, self.fold_num)
        # whole_set = np.array(whole_set)
        self.train_data, self.train_label, self.test_data, self.test_label = whole_set[0]
        self.test_data, self.test_label = valence_class(self.test_data, self.test_label, self.class_num)
        if self.sampling_option != "None":
            self.train_data, self.train_label = over_sampling(self.train_data, self.train_label, self.sampling_option)
        self.check_patch_shape(patch_size=self.patch_size)
        # if self.noise_augment:
        #     self.augment_noise()

    def check_image_shape(self):
        sample_image_path = self.train_data[0]
        print('checking image shape... : ', sample_image_path)
        self.input_image_shape = check_image_shape(sample_image_path)
        print('input image shape : ',self.input_image_shape)

    def check_patch_shape(self, patch_size):
        sample_image_path1, sample_image_path2 = self.train_data[0].split(',')
        # sample_image_path1, sample_image_path2 = self.train_data[0]
        print('checking patch image shape... : ', sample_image_path1)
        # self.input_image_shape = check_image_shape(sample_image_path1)
        self.input_image_shape = (patch_size,patch_size,patch_size)
        print('input patch shape : ',self.input_image_shape)

    def noise_addition(self, data):
        return gaussian_noise_layer(data, std=0.01)

    ##################################################################################
    # validation
    ##################################################################################
    # def try_all_fold(self):
    #     result_list = _validation.try_all_fold(self)
    #     _validation.save_results(self, result_list)
    #
    # def BayesOptimize(self, init_lr_log, w_stddev_log):
    #     _BO.BayesOptimize(init_lr_log, w_stddev_log)
    ##################################################################################
    # Model
    ##################################################################################
    def select_network(self):
        if self.model_name == 'simple':
            self.network = SimpleNet
        if self.model_name == 'siam':
            self.network = Siamese
            # self.model_name = self.cnn_simple_patch
        # if args.neural_net == 'basic':
        #     self.model_name = self.neural_net_basic
        # if args.neural_net == 'simple':
        #     self.model_name = self.neural_net_simple
        else:
            self.network = None

    def build_model(self):
        s1,s2,s3 = self.input_image_shape
        self.input = tf.placeholder(tf.float32, [None, s1*2 ,s2, s3, 1], name='inputs')
        print(self.input.shape)
        # self.label = tf.placeholder(tf.float32, [None, self.class_num], name='targets')
        self.label = tf.placeholder(tf.int32, [None], name='targets')
        # self.label_onehot = tf.stop_gradient(onehot(self.label, self.class_num))
        self.label_onehot = onehot(self.label, self.class_num)

        self.select_network()
        self.my_model = self.network(weight_initializer=tf.truncated_normal_initializer,
                                  activation=tf.nn.relu,
                                  class_num=self.class_num,
                                  patch_size=s1,
                                  patch_num=2)

        self.logits = self.my_model.model(self.input)
        self.pred = tf.argmax(self.logits,1)
        self.accur = accuracy(self.logits, self.label_onehot) // 1
        with tf.name_scope('Loss'):
            self.loss = classifier_loss(self.loss_function, predictions=self.logits, targets=self.label_onehot)

        t_vars = tf.trainable_variables()
        # if i need to access specific variables, use below line
        # vars = [var for var in t_vars if 'cnn' in var.name]
        # for var in t_vars:
        #     tf.summary.image(var.name, var.)
        #     print(var)
        #     tf.summary.histogram(var.name, var)
        # should apply learning rate decay
        with tf.name_scope('learning_rate_decay'):
            start_lr = self.learning_rate
            global_step = tf.Variable(0, trainable=False)
            total_learning = self.epoch
            # lr = tf.train.exponential_decay(start_lr, global_step,total_learning,0.99999, staircase=True)
            lr = tf.train.exponential_decay(start_lr, global_step, decay_steps=self.epoch//100, decay_rate=.96, staircase=True)

        with tf.variable_scope('optimizer'):
            self.optim = tf.train.AdamOptimizer(lr).minimize(self.loss)
            # self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            #self.d_optim = tf.train.AdagradOptimizer(d_lr).minimize(self.loss, var_list=d_vars)

        """ Summary """
        tf.summary.scalar("loss__", self.loss)
        tf.summary.scalar('accuracy', self.accur)
        self.merged_summary = tf.summary.merge_all()

    ##################################################################################
    # Train
    ##################################################################################
    def test_data_read(self):
        print("test data reading ... not training ...")
        self.next_element, self.iterator = get_patch_dataset(self.train_data, self.train_label, self.batch_size)
        self.sess.run(self.iterator.initializer)
        for i in range(3):
            train_data, train_label = self.sess.run(self.next_element)
            print(train_label)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        # graph inputs for visualize training results
        # saver to save model
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter('../' + self.log_dir + '/' + self.model_dir +'_train', self.sess.graph)
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
        # set training data
        print("set training and testing dataset ... ")
        self.next_element, self.iterator = get_patch_dataset(
            self.train_data,
            self.train_label,
            self.args.buffer_scale,
            self.args.mask,
            self.batch_size)

        self.test_element, self.test_iterator = get_patch_dataset(
            self.test_data,
            self.test_label,
            self.args.buffer_scale,
            self.args.mask,
            len(self.test_label))

        self.sess.run(self.iterator.initializer)
        self.sess.run(self.test_iterator.initializer)
        test_data_ts, test_label_ts = self.sess.run(self.test_element)
        test_feed_dict = {
            self.input: test_data_ts,
            self.label: test_label_ts
        }
        '''
        next_element, iterator = get_patch_dataset(train_data, train_label, args.buffer_scale, is_mask, batch)
        sess.run(iterator.initializer)
        test_element, test_iterator = get_patch_dataset(val_data, val_label, args.buffer_scale, is_mask, len(val_label))
        sess.run(test_iterator.initializer)
        val_data_ts, test_label_ts = sess.run(test_element)

        '''
        print("start training ... ")
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.iteration):

                train_data, train_label = self.sess.run(self.next_element)
                train_feed_dict = {
                    self.input : train_data,
                    self.label : train_label
                }

                _, train_merged_summary, loss, logits, pred, accur = self.sess.run(
                    [self.optim, self.merged_summary, self.loss, self.logits, self.pred, self.accur],
                    feed_dict=train_feed_dict)

                self.train_writer.add_summary(train_merged_summary, global_step=counter)
                # self.train_writer.add_summary(train_merged_summary, global_step=counter)

                if epoch % self.print_freq == 0:

                    test_merged_summary, test_accur = self.sess.run(
                        [self.merged_summary, self.accur],
                        feed_dict=test_feed_dict)

                    print("Epoch: [{}/{}] [{}/{}], loss: {}, accur: {}" \
                          .format(epoch, self.epoch, idx, self.iteration,loss, accur))
                    label_sample = train_label[:5]
                    train_sample = logits[:5] // 0.01
                    for i,j in zip(label_sample,train_sample):
                        print("label : {} , pred : {}".format(i,j))
                    print('test accur : {}'.format(test_accur))

                    # test_accur, test_summary = self.test(counter)
                    self.valid_accur.append(test_accur)
                    self.train_accur.append(accur)
                    print('=' * 100)
                # if epoch % self.summary_freq == 0:
                #     self.train_writer.add_summary(merged_summary, global_step=counter)
                #     if self.investigate_validation:
                #         self.test(counter)
            counter+=1

        print(self.train_accur)
        print(self.valid_accur)
        return np.max(self.valid_accur)

    def test(self, counter):
        test_feed_dict = {
            self.input: self.test_data_ts,
            self.label: self.test_label_ts
        }
        # tf.global_variables_initializer().run()
        loss, accur, pred, merged_summary_str = self.sess.run([self.loss, self.accur, self.pred, self.merged_summary], feed_dict=test_feed_dict)

        # self.test_writer.add_summary(merged_summary_str, counter)
        if self.investigate_validation:
            pass
        else:
            print("Test result => accur : {}, loss : {}".format(accur, loss))
            print("label : {}".format(self.test_label))
            print("pred  : {}".format(pred))
        return accur, merged_summary_str
    # def test(self, counter):
    #     test_feed_dict = {
    #         self.input: self.test_data,
    #         self.label: self.test_label
    #     }
    #     # tf.global_variables_initializer().run()
    #     loss, accur, pred, merged_summary_str = self.sess.run([self.loss, self.accur, self.pred, self.merged_summary], feed_dict=test_feed_dict)
    #
    #     # self.test_writer.add_summary(merged_summary_str, counter)
    #     if self.investigate_validation:
    #         pass
    #     else:
    #         print("Test result => accur : {}, loss : {}".format(accur, loss))
    #         print("pred : {}".format(self.test_label))
    #         print("pred : {}".format(pred))
    #     return accur, merged_summary_str

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
                train_feed_dict = {
                    self.input: self.train_data,
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
            '/home/soopil/Desktop/github/brainMRI_classification/regression_result/chosun_MRI_excel_logistic_regression_result_' \
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