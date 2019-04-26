import sys
import tensorflow as tf
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification')
sys.path.append('/home/sp/PycharmProjects/brainMRI_classification/NeuralNet')
# from NeuralNet.neuralnet_ops import *
import NeuralNet.NN_validation as _validation
import NeuralNet.NN_BO as _BO
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
            self.model_name = self.neural_net_simple
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
        # self.batch_size = args.batch_size
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

    def set_model(self, model):
        pass
    ##################################################################################
    # Custom Operation
    ##################################################################################
    def sample_save(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet", reuse=reuse):
            ch = 64
            x = conv(x, channels=ch, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv')
            x = lrelu(x, 0.2)
            if is_print:
                print(x.shape)
                print('repeat layer : {}'.format(self.layer_num))
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
            if is_print:print('repeat layer : {}'.format(self.layer_num))
            # for i in range(self.layer_num // 2, self.layer_num):
            for i in range(12):
                x = resblock(x, ch, use_bias=True,sn=False, scope='resblock'+str(i))
            if is_print:print(x.shape)
            x = conv(x, channels=4, stride=1, sn=self.sn, use_bias=False, scope='D_logit')
            if is_print:print(x.shape)
            # assert False
            return x

    def self_attention_nn(self, x, ch, scope='attention', reuse=False):
        assert ch//8 >= 1
        with tf.variable_scope(scope, reuse=reuse):
            ch_ = ch // 8
            if ch_ == 0: ch_ = 1
            f = self.fc_layer(x, ch_, 'f_nn') # [bs, h, w, c']
            g = self.fc_layer(x, ch_, 'g_nn') # [bs, h, w, c']
            h = self.fc_layer(x, ch, 'h_nn') # [bs, h, w, c]
            # N = h * w
            s = tf.matmul(g, f, transpose_b=True) # # [bs, N, N]
            beta = tf.nn.softmax(s, axis=-1)  # attention map
            o = tf.matmul(beta, h) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            print(o.shape, s.shape, f.shape, g.shape, h.shape)
            # o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            x = gamma * o + x
        return x

    def attention_nn(self, x, ch, scope='attention', reuse=False):
        assert ch//8 >= 1
        with tf.variable_scope(scope, reuse=reuse):
            i = self.fc_layer(x, ch, 'fc_1')
            i = self.fc_layer(i, ch//4, 'fc_2')
            i = self.fc_layer(i, ch//8, 'fc_3')
            i = self.fc_layer(i, ch//4, 'fc_4')
            i = self.fc_layer(i, ch, 'fc_5')
            o = tf.nn.softmax(i, axis=-1)  # attention map
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            print(i.shape, o.shape)
            # o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
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
    # def attention(self, x, ch, sn=False, scope='attention', reuse=False):
    #     with tf.variable_scope(scope, reuse=reuse):
    #         ch_ = ch // 8
    #         if ch_ == 0: ch_ = 1
    #         f = conv(x, ch_, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
    #         g = conv(x, ch_, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
    #         h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]
    #
    #         # N = h * w
    #         s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
    #
    #         beta = tf.nn.softmax(s, axis=-1)  # attention map
    #
    #         o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    #         gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    #         print(o.shape, s.shape, f.shape, g.shape, h.shape)
    #
    #         o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
    #         x = gamma * o + x
    #     return x

    ##################################################################################
    # Neural Network Model
    ##################################################################################
    def neural_net_simple(self, x, is_training=True, reuse=False):
        layer_num = 3
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet", reuse=reuse):
            x = self.fc_layer(x, 512, 'fc_input_1')
            x = self.fc_layer(x, 1024, 'fc_input_2')
            for i in range(layer_num):
                x = self.fc_layer(x, 1024, 'fc'+str(i))
            x = self.fc_layer(x, 512, 'fc_1')
            x = self.fc_layer(x, 256, 'fc_fin')
            # x = self.fc_layer(x, self.class_num, 'fc_last')
            x = fully_connected(x, self.class_num,\
                                weight_initializer=self.weight_initializer, use_bias=True, scope='fc_last')
            # tf.summary.histogram('last_active', x)
            return x

    def neural_net_attention(self, x, is_training=True, reuse=False):
        layer_num = 2
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet", reuse=reuse):
            x = self.fc_layer(x, 1024, 'fc_en_1')
            x = self.fc_layer(x, 512, 'fc_en_2')
            x = self.fc_layer(x, 256, 'fc_en_3')
            x = self.fc_layer(x, 256, 'fc_en_4')
            x = self.attention_nn(x, 256)
            x = self.fc_layer(x, 256, 'fc_de_1')
            x = self.fc_layer(x, 512, 'fc_de_2')
            x = self.fc_layer(x, 512, 'fc_de_3')
            x = self.fc_layer(x, 256, 'fc_de_4')
            x = fully_connected(x, self.class_num, \
                                weight_initializer=self.weight_initializer, use_bias=True, scope='fc_last')
            # x = self.fc_layer(x, self.class_num, 'fc_last')
            # tf.summary.histogram('last_active', x)
            return x

    def neural_net_attention_often(self, x, is_training=True, reuse=False):
        layer_num = 2
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet", reuse=reuse):
            en_dim = 256
            de_dim = 256
            x = self.fc_layer(x, 1024, 'fc_en_1')
            x = self.fc_layer(x, 512, 'fc_en_2')
            x = self.fc_layer(x, 256, 'fc_en_3')
            x = self.attention_nn(x, 256, 'attention_1')
            x = self.fc_layer(x, 256, 'bridge_1')
            x = self.attention_nn(x, 256, 'attention_2')
            x = self.fc_layer(x, 256, 'bridge_2')
            x = self.attention_nn(x, 256, 'attention_3')
            x = self.fc_layer(x, 512, 'fc_de_1')
            x = self.fc_layer(x, 256, 'fc_de_2')
            x = self.fc_layer(x, 128, 'fc_de_3')
            x = fully_connected(x, self.class_num, \
                                weight_initializer=self.weight_initializer, use_bias=True, scope='fc_last')
            # x = self.fc_layer(x, self.class_num, 'fc_last')
            # tf.summary.histogram('last_active', x)
            return x

    def neural_net_self_attention(self, x, is_training=True, reuse=False):
        layer_num = 2
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet", reuse=reuse):
            x = self.fc_layer(x, 512, 'fc_input_1')
            x = self.fc_layer(x, 256, 'fc_input_2')
            x = self.fc_layer(x, 128, 'fc_input_3')
            x = self.self_attention_nn(x, 128)
            x = self.fc_layer(x, 256, 'fc_input_4')
            x = self.fc_layer(x, 256, 'fc_input_5')
            x = self.fc_layer(x, 256, 'fc_input_6')
            x = fully_connected(x, self.class_num, \
                                weight_initializer=self.weight_initializer, use_bias=True, scope='fc_last')

            # x = self.fc_layer(x, self.class_num, 'fc_last')
            # tf.summary.histogram('last_active', x)
            return x

    def neural_net_basic(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)

        with tf.variable_scope("neuralnet", reuse=reuse):
            # x = fully_connected(x, self.class_num, use_bias=True, scope='fc2')
            # x = lrelu(x, 0.1)
            x = self.fc_layer(x, 512, 'fc1')
            x = self.fc_layer(x, self.class_num, 'fc2')
            return x

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
        return gaussian_noise_layer(data, std=0.01)

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

    def BayesOptimize(self):
        bayes_optimizer = BayesianOptimization(
            f=self.BO_train_and_validate,
            pbounds={
                # 78 <= -1.2892029132535314,-1.2185073691640054
                # 85 <= -1.2254855784556566, -1.142561108840614}}
                'init_lr_log': (-2.0,-1.0),
                'w_stddev_log': (-2.0,-1.0)
            },
            random_state=0,
            # verbose=2
        )
        bayes_optimizer.maximize(
            init_points=5,
            n_iter=40,
            acq='ei',
            xi=0.01
        )
        BO_results = []
        BO_results.append('\n\t\t<<< class option : {} >>>\n' .format(self.class_option))
        BO_result_file_name = "BO_result/BayesOpt_results"\
                              + str(time.time()) + '_' + self.class_option
        fd = open(BO_result_file_name, 'a+t')
        for i, ressult in enumerate(bayes_optimizer.res):
            BO_results.append('Iteration {}:{}\n'.format(i, ressult))
            print('Iteration {}: {}'.format(i, ressult))
            fd.writelines('Iteration {}:{}\n'.format(i, ressult))
        BO_results.append('Final result: {}\n'.format(bayes_optimizer.max))
        fd.writelines('Final result: {}\n'.format(bayes_optimizer.max))
        print('Final result: {}\n'.format(bayes_optimizer.max))
        fd.close()
    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        """ Graph Input """
        self.input = tf.placeholder(tf.float32, [None, self.input_feature_num], name='inputs')
        # self.label = tf.placeholder(tf.float32, [None, self.class_num], name='targets')
        self.label = tf.placeholder(tf.int32, [None], name='targets')
        self.label_onehot = onehot(self.label, self.class_num)
        # output of D for real images
        print(self.input)

        self.logits = self.model_name(self.input, reuse=False)
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
        vars = [var for var in t_vars if 'neuralnet' in var.name]

        # optimizers
        # should apply learning rate decay
        start_lr = self.learning_rate
        global_step = tf.Variable(0, trainable=False)
        total_learning = self.epoch
        lr = tf.train.exponential_decay(start_lr, global_step,
                                        decay_steps=self.epoch//100,
                                        decay_rate=.96,
                                        staircase=True)
        # self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.optim = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
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