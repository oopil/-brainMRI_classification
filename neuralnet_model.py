import time
from neuralnet_ops import *
from neuralnet_utils import *
from data_merge import *

class NeuralNet(object):
    def __init__(self, sess, args):
        self.model_name = "NeuralNet"  # name for checkpoint
        self.sess = sess
        self.excel_path = args.excel_path
        self.base_folder_path = args.base_folder_path

        self.diag_type = args.diag_type
        self.excel_option = args.excel_option
        self.test_num = args.test_num
        self.fold_num = args.fold_num
        self.is_split_by_num = args.is_split_by_num
        self.sampling_option = args.sampling_option
        self.learning_rate = args.lr

        self.class_option = args.class_option
        self.class_option_index = args.class_option_index
        class_split = self.class_option.split('vs')
        self.class_num = len(class_split)

        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.epoch = args.epoch
        self.iteration = args.iter
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        # self.iteration = args.iteration
        # self.batch_size = args.batch_size
        self.is_print = True

        print()
        print("##### Information #####")
        print("# epoch : ", self.epoch)

    ##################################################################################
    # Discriminator
    ##################################################################################
    def neural_net(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet", reuse=reuse):
            # x = fully_connected(x, 512, use_bias=True, scope='fc1')
            # x = lrelu(x, 0.1)
            # x = fully_connected(x, self.class_num, use_bias=True, scope='fc2')
            # x = lrelu(x, 0.1)
            x = self.fc_layer(x, 512, 'fc1')
            x = self.fc_layer(x, self.class_num, 'fc2')
            return x

    def fc_layer(self, x, ch, scope):
        x = fully_connected(x, ch, use_bias=True, scope=scope)
        x = lrelu(x, 0.1)
        return x

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

    def attention(self, x, ch, sn=False, scope='attention', reuse=False):
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
    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        """ Graph Input """
        self.input = tf.placeholder(tf.float32, [None, self.input_feature_num], name='inputs')
        # self.label = tf.placeholder(tf.float32, [None, self.class_num], name='targets')
        self.label = tf.placeholder(tf.int32, [None], name='targets')
        self.label_onehot = onehot(self.label, self.class_num)
        """ Loss Function """
        # output of D for real images
        print(self.input)
        pred = self.neural_net(self.input)
        # get loss for discriminator
        self.loss = classifier_loss('normal', predictions=pred, targets=self.label_onehot)
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if 'neuralnet' in var.name]

        # optimizers
        # should apply learning rate decay
        start_lr = self.learning_rate
        global_step = tf.Variable(0, trainable=False)
        total_learning = self.epoch
        lr = tf.train.exponential_decay(start_lr, global_step,total_learning,0.99999, staircase=True)
        self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        # self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.loss, var_list=d_vars)
        # self.d_optim = tf.train.AdamOptimizer(d_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.loss, var_list=d_vars)
        #self.d_optim = tf.train.AdagradOptimizer(d_lr).minimize(self.loss, var_list=d_vars)

        """ Summary """
        self.sum = tf.summary.scalar("loss", self.loss)

    # def read_excel_data(self, base_folder_path, excel_path):
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
        self.input_feature_num = len(self.train_data[0])
        # print(onehot(self.train_label,self.class_num))
        # print(type(self.train_label))
        # print(type(np.array(self.train_label)))

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
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

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

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.iteration):
                #---------------------------------------------------
                train_feed_dict = {
                    self.input : self.train_data,
                    self.label : self.train_label
                }
                _, summary_str, loss = self.sess.run([self.optim, self.sum, self.loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                print("Epoch: [%2d/%2d] [%5d/%5d] time: %4.4f, loss: %.8f, loss: %.8f" \
                      % (epoch, self.epoch, idx, self.iteration, time.time() - start_time, loss, loss))

            #     # save training results for every 300 steps
            #     if np.mod(idx+1, self.print_freq) == 0:
            #         samples = self.sess.run(self.fake_images, feed_dict={self.inputs_sketch: self.sample_sketch})
            #         tot_num_samples = min(self.sample_num, self.batch_size)
            #         manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
            #         manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
            #         # save_images(inputs_sketches[:manifold_h * manifold_w, :, :, :],
            #         #             [manifold_h, manifold_w],
            #         #             './' + self.sample_dir + '/' + self.model_name + '_sketch_{:02d}_{:05d}.png'.format(epoch, idx+1))
            #         save_images(samples[:manifold_h * manifold_w, :, :, :],
            #                     [manifold_h, manifold_w],
            #                     './' + self.sample_dir + '/' + self.model_name + '_train_{:02d}_{:05d}.png'.format(epoch, idx+1))
            #
            #     if np.mod(idx+1, self.save_freq) == 0:
            #         self.save(self.checkpoint_dir, counter)
            #
            # # After an epoch, start_batch_id is set to zero
            # # non-zero value is only for the first epoch after loading pre-trained model
            # start_batch_id = 0
            #
            # # save model
            # self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}".format(
            self.model_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

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

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.sample_dir + '/' + self.model_name + '_epoch%02d' % epoch + '_visualize.png')

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        for i in range(self.test_num) :
            z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                        [image_frame_dim, image_frame_dim],
                        result_dir + '/' + self.model_name + '_test_{}.png'.format(i))



'''
keep_prob = 0.9 # 0.9
learning_rate = 0.01
epochs = 2000
print_freq = 200
save_freq = 200

layer = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]
    layer_last = 256
    with tf.name_scope("FCN"):
        #defining the network
        with tf.name_scope("layer1"):
            l1 = fully_connected_layer(inputs, feature_num, layer[0], keep_prob)
            l2 = fully_connected_layer(l1, layer[0], layer[1], keep_prob)
            l3 = fully_connected_layer(l2, layer[1], layer[2], keep_prob) + l2

        with tf.name_scope("layer2"):
            l4 = fully_connected_layer(l3, layer[2], layer[3], keep_prob)
            l5 = fully_connected_layer(l4, layer[3], layer[4], keep_prob)
            l6 = fully_connected_layer(l5, layer[4], layer[5], keep_prob)
            l7 = fully_connected_layer(l6, layer[5], layer[6], keep_prob)
            l8 = fully_connected_layer(l7, layer[6], layer[7], keep_prob) + l4
        with tf.name_scope("layer3"):
            l9 = fully_connected_layer(l8, layer[2], layer[3], keep_prob)
            l10 = fully_connected_layer(l9, layer[3], layer[4], keep_prob)
            l11 = fully_connected_layer(l10, layer[4], layer[5], keep_prob)
            l12 = fully_connected_layer(l11, layer[5], layer[6], keep_prob)
            l13 = fully_connected_layer(l12, layer[6], layer[7], keep_prob) + l9
        with tf.name_scope("layer4"):
            l14 = fully_connected_layer(l13, layer[2], layer[3], keep_prob)
            l15 = fully_connected_layer(l14, layer[3], layer[4], keep_prob)
            l16 = fully_connected_layer(l15, layer[4], layer[5], keep_prob)
            l17 = fully_connected_layer(l16, layer[5], layer[6], keep_prob)
            l18 = fully_connected_layer(l17, layer[6], layer[7], keep_prob) + l14
        with tf.name_scope("layer5"):
            l19 = fully_connected_layer(l18, layer[2], layer[2], keep_prob)
            l20 = fully_connected_layer(l19, layer[3], layer[3], keep_prob)
            l21 = fully_connected_layer(l20, layer[4], layer[4], keep_prob)
            l22 = fully_connected_layer(l21, layer[5], layer[5], keep_prob)
            l23 = fully_connected_layer(l22, layer[6], layer[6], keep_prob) + l19
            l24 = fully_connected_layer(l23, layer[6], layer_last, keep_prob)
            l_fin = fully_connected_layer(l24, layer_last, class_num, activation=None, keep_prob=keep_prob)

        #defining special parameter for our predictions - later used for testing
        predictions = tf.nn.sigmoid(l_fin)

'''
'''
keep_prob = 0.9 # 0.9

learning_rate = 0.01
epochs = 2000
print_freq = 200
save_freq = 200

    layer = [512, 1024, 2048, 1024]
    layer_last = 256
    with tf.name_scope("FCN"):
        #defining the network
        with tf.name_scope("layer1"):
            l1 = fully_connected_layer(inputs, feature_num, layer[0], keep_prob)
            l2 = fully_connected_layer(l1, layer[0], layer[0], keep_prob)# + l1
            l3 = fully_connected_layer(l2, layer[0], layer[0], keep_prob) + l2

        with tf.name_scope("layer2"):
            l4 = fully_connected_layer(l3, layer[0], layer[0], keep_prob)# + l3
            l5 = fully_connected_layer(l4, layer[0], layer[0], keep_prob)# + l4# + l3
            l6 = fully_connected_layer(l5, layer[0], layer[0], keep_prob)# + l5# + l4 + l3
            l7 = fully_connected_layer(l6, layer[0], layer[0], keep_prob)# + l6# + l5 + l4 + l3
            l8 = fully_connected_layer(l7, layer[0], layer[0], keep_prob) + l5# + l7# + l6 + l5 + l4 + l3
        with tf.name_scope("layer3"):
            l9 = fully_connected_layer(l8, layer[0], layer[1], keep_prob)# + l8
            l10 = fully_connected_layer(l9, layer[1], layer[1], keep_prob)# + l9# + l8
            l11 = fully_connected_layer(l10, layer[1], layer[1], keep_prob)# + l10# + l9 + l8
            l12 = fully_connected_layer(l11, layer[1], layer[1], keep_prob)# + l11# + l10 + l9 + l8
            l13 = fully_connected_layer(l12, layer[1], layer[1], keep_prob) + l10# + l12# + l11 + l10 + l9 + l8
        with tf.name_scope("layer4"):
            l14 = fully_connected_layer(l13, layer[1], layer[2], keep_prob)# + l13
            l15 = fully_connected_layer(l14, layer[2], layer[2], keep_prob)# + l14# + l13
            l16 = fully_connected_layer(l15, layer[2], layer[2], keep_prob)# + l15# +l14 + l13
            l17 = fully_connected_layer(l16, layer[2], layer[2], keep_prob)# + l16# + l15 + l14 + l13
            l18 = fully_connected_layer(l17, layer[2], layer[2], keep_prob) + l15# + l17# + l16 + l15 + l14 + l13
        with tf.name_scope("layer5"):
            l19 = fully_connected_layer(l18, layer[2], layer[3], keep_prob)# + l18
            l20 = fully_connected_layer(l19, layer[3], layer[3], keep_prob)# + l19# + l18
            l21 = fully_connected_layer(l20, layer[3], layer[3], keep_prob)# + l20# + l19 + l18
            l22 = fully_connected_layer(l21, layer[3], layer[3], keep_prob)# + l21# + l20 + l19 + l18
            l23 = fully_connected_layer(l22, layer[3], layer[3], keep_prob) + l20# + l22# + l21 + l20 + l19 + l18
            l24 = fully_connected_layer(l23, layer[3], layer_last, keep_prob)# + l23
            l_fin = fully_connected_layer(l24, layer_last, class_num, activation=None, keep_prob=keep_prob)

        #defining special parameter for our predictions - later used for testing
        predictions = tf.nn.sigmoid(l_fin)

'''