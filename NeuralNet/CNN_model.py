import sys

sys.path.append('/home/sp/PycharmProjects/brainMRI_classification')
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification/NeuralNet')
# from NeuralNet.neuralnet_ops import *
import NeuralNet.NN_validation as _validation
import NeuralNet.NN_BO as _BO
import tensorflow as tf
from NeuralNet.NN_ops import *
from NeuralNet.CNN_data import *
from data_merge import *

class ConvNeuralNet:
    def __init__(self, sess, args):
        self.model_name = "CNN"  # name for checkpoint
        self.sess = sess
        self.excel_path = args.excel_path
        self.base_folder_path = args.base_folder_path
        self.result_file_name = args.result_file_name

        if args.neural_net == 'simple':
            self.model_name = self.cnn_simple
        # if args.neural_net == 'basic':
        #     self.model_name = self.neural_net_basic
        # if args.neural_net == 'simple':
        #     self.model_name = self.neural_net_simple

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
        self.is_print = True
        self.args = args

        print()
        print("##### Information #####")
        for i, arg in enumerate(vars(args)):
            print(i, arg, getattr(args, arg))
        # assert False
        # print("# epoch : ", self.epoch)


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
    # Custom Operation
    ##################################################################################
    def sample_save(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("cnn", reuse=reuse):
            ch = 64
            x = conv(x, channels=ch, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv')
            x = lrelu(x, 0.2)
            for i in range(self.layer_num // 2):
                x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm' + str(i))
                x = lrelu(x, 0.2)
                ch = ch * 2
            # Self Attention
            x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse)
            if is_print:
                print('attention !')
                print(x.shape)
                print('repeat layer : {}'.format(self.layer_num))
            # for i in range(self.layer_num // 2, self.layer_num):
            for i in range(12):
                x = resblock(x, ch, use_bias=True,sn=False, scope='resblock'+str(i))
            x = conv(x, channels=4, stride=1, sn=self.sn, use_bias=False, scope='D_logit')
            # assert False
            return x

    def attention_nn(self, x, ch, sn=False, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            ch_ = ch // 8
            if ch_ == 0: ch_ = 1
            f = conv(x, ch_, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
            g = conv(x, ch_, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
            h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            print(o.shape, s.shape, f.shape, g.shape, h.shape)

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            x = gamma * o + x
        return x

    def fc_layer(self, x, ch, scope):
        with tf.name_scope(scope):
            x = fully_connected(x, ch, weight_initializer=self.weight_initializer, \
                                use_bias=True, scope=scope)
            # tf.summary.histogram('active', x)
            # x = lrelu(x, 0.1)
            x = relu(x, scope=scope)
        return x
    ##################################################################################
    # Convolutional Neural Network Model
    ##################################################################################
    def cnn_simple(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print('input shape : {}'.format(x.shape))

        with tf.variable_scope("cnn", reuse=reuse):
            ch = 128
            x = lrelu(conv3d(x, ch, ks=4, s=(2, 2, 2), name='layer1'))
            # h0 is (128 x 128 x self.df_dim)
            x = lrelu(instance_norm(conv3d(x, ch, ks=4, s=(2, 2, 2), name='layer2')))
            # h1 is (64 x 64 x self.df_dim*2)
            x = lrelu(instance_norm(conv3d(x, ch, ks=4, s=(2, 2, 2), name='layer3')))
            # h2 is (32x 32 x self.df_dim*4)
            x = lrelu(instance_norm(conv3d(x, ch, ks=4, s=(2, 2, 2), name='layer4')))

        with tf.variable_scope("fcn", reuse=reuse):
            x = flatten(x)
            x = self.fc_layer(x, 512, 'fc1')
            x = self.fc_layer(x, self.class_num, 'fc2')

            return x

    def neural_net_basic(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print('input shape : {}'.format(x.shape))

        with tf.variable_scope("cnn", reuse=reuse):
            # x = fully_connected(x, self.class_num, use_bias=True, scope='fc2')
            # x = lrelu(x, 0.1)
            x = self.fc_layer(x, 512, 'fc1')
            x = self.fc_layer(x, self.class_num, 'fc2')
            return x

    ##################################################################################
    # Dataset
    ##################################################################################
    def read_cnn_data(self):
        # None RANDOM ADASYN SMOTE SMOTEENN SMOTETomek BolderlineSMOTE
        sampling_option_str = 'None RANDOM SMOTE SMOTEENN SMOTETomek BolderlineSMOTE'  # ADASYN
        sampling_option_split = sampling_option_str.split(' ')
        whole_set = CNN_dataloader(self.base_folder_path, self.diag_type, self.class_option,\
                                  self.excel_path, self.test_num, self.fold_num, self.is_split_by_num)
        # whole_set = np.array(whole_set)
        self.train_data, self.train_label, self.test_data, self.test_label = whole_set[0]
        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)
        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)

        self.test_data, self.test_label = valence_class(self.test_data, self.test_label, self.class_num)
        self.train_data, self.train_label = over_sampling(self.train_data, self.train_label, self.sampling_option)

        self.check_image_shape()

        # if self.noise_augment:
        #     self.augment_noise()

    def check_image_shape(self):
        sample_image_path = self.train_data[0]
        self.input_image_shape = check_image_shape(sample_image_path)
        print('input image shape : ',self.input_image_shape)

    def augment_noise(self):
        self.train_data, self.train_label = \
            augment_noise(self.train_data, self.train_label, self.noise_augment)

    ##################################################################################
    # validation
    ##################################################################################
    def try_all_fold(self):
        result_list = _validation.try_all_fold(self)
        _validation.save_results(self, result_list)

    def BayesOptimize(self, init_lr_log, w_stddev_log):
        _BO.BayesOptimize(init_lr_log, w_stddev_log)
    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        """ Graph Input """
        s1,s2,s3 = self.input_image_shape
        self.input = tf.placeholder(tf.float32, [None, s1,s2,s3, 1], name='inputs')
        print(self.input.shape)
        # self.label = tf.placeholder(tf.float32, [None, self.class_num], name='targets')
        self.label = tf.placeholder(tf.int32, [None], name='targets')
        self.label_onehot = onehot(self.label, self.class_num)
        # output of D for real images
        print(self.input)
        self.logits = self.model_name(self.input, reuse=tf.AUTO_REUSE)
        self.pred = tf.argmax(self.logits,1)
        self.accur = accuracy(self.logits, self.label_onehot) //1

        # get loss for discriminator
        """ Loss Function """
        with tf.name_scope('Loss'):
            # self.loss = classifier_loss('normal', predictions=self.logits, targets=self.label_onehot)
            self.loss = classifier_loss(self.loss_function, predictions=self.logits, targets=self.label_onehot)
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if 'cnn' in var.name]

        # optimizers
        # should apply learning rate decay
        with tf.name_scope('learning_rate'):
            start_lr = self.learning_rate
            global_step = tf.Variable(0, trainable=False)
            total_learning = self.epoch
            lr = tf.train.exponential_decay(start_lr, global_step,total_learning,0.99999, staircase=True)

        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.loss, var_list=d_vars)
        # self.d_optim = tf.train.AdamOptimizer(d_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.loss, var_list=d_vars)
        #self.d_optim = tf.train.AdagradOptimizer(d_lr).minimize(self.loss, var_list=d_vars)

        """ Summary """
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar('accuracy', self.accur)
        self.merged_summary = tf.summary.merge_all()

    ##################################################################################
    # Train
    ##################################################################################
    def test_data_read(self):
        self.next_element, self.iterator = get_dataset(self.train_data, self.train_label, self.batch_size)
        self.sess.run(self.iterator.initializer)
        train_data, train_label = self.sess.run(self.next_element)
        print(train_label)

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
        # set training data
        self.next_element, self.iterator = get_dataset(self.train_data, self.train_label, self.batch_size)
        self.sess.run(self.iterator.initializer)
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.iteration):
                train_data, train_label = self.sess.run(self.next_element)
                #---------------------------------------------------
                train_feed_dict = {
                    self.input : train_data,
                    self.label : train_label
                }
                _, merged_summary_str, loss, pred, accur = self.sess.run( \
                    [self.optim, self.merged_summary, self.loss, self.pred, self.accur], \
                    feed_dict=train_feed_dict)
                # self.train_writer.add_summary(merged_summary_str, global_step=counter)

                # if epoch % self.print_freq == 0:
                #     print("Epoch: [{}/{}] [{}/{}], loss: {}, accur: {}" \
                #           .format(epoch, self.epoch, idx, self.iteration,loss, accur))
                #     # print("Epoch: [%2d/%2d] [%5d/%5d] time: %4.4f, loss: %.8f" \
                #     #       % (epoch, self.epoch, idx, self.iteration, time.time() - start_time, loss))
                #     # print("pred : {}".format(self.train_label))
                #     # print("pred : {}".format(pred))
                #
                #     test_accur, test_summary = self.test(counter)
                #     self.valid_accur.append(test_accur)
                #     self.train_accur.append(accur)
                #     print('=' * 100)
                #
                # if epoch % self.summary_freq == 0:
                #     self.train_writer.add_summary(merged_summary_str, global_step=counter)
                #     if self.investigate_validation:
                #         self.test(counter)
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