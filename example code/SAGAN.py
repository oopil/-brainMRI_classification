import time
from NeuralNet.neuralnet_ops import *
from utils import *
from dataloader import *


class SAGAN(object):

    def __init__(self, sess, args):
        self.model_name = "SAGAN"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.load_size = args.load_size
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_size = args.img_size

        self.resblock_num = args.resblock_num
        self.is_print = True
        self.is_attention = args.attention
        # self.is_print = False
        """ Generator """
        if args.gen == 'generator_res':
            self.generator = self.generator_res
        elif args.gen == 'generator':
            self.generator = self.generator
        elif args.gen == 'generator_attention':
            self.generator = self.generator_attention()
        elif args.gen == 'generator_attent':
            self.generator = self.generator_attent
        elif args.gen == 'generator_save':
            self.generator = self.generator_save

        self.layer_num = int(np.log2(self.img_size)) - 3
        self.z_dim = args.z_dim  # dimension of noise-vector
        self.up_sample = args.up_sample
        self.gan_type = args.gan_type

        """ Discriminator """
        self.n_critic = args.n_critic
        self.sn = args.sn
        self.ld = args.ld


        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num


        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.custom_dataset = True

        if self.dataset_name == 'mnist' :
            self.c_dim = 1
            self.data = load_mnist(size=self.img_size)

        elif self.dataset_name == 'cifar10' :
            self.c_dim = 3
            self.data = load_cifar10(size=self.img_size)

        else :
            self.c_dim = 3
            self.data_img_l, self.data_sketch_l = LoadData(load_size)
            self.data = load_data(dataset_name=self.dataset_name, size=self.img_size)
            self.custom_dataset = True


        self.dataset_num = len(self.data)

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# generator layer : ", self.layer_num)
        print("# upsample conv : ", self.up_sample)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.layer_num)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, z, is_training=True, reuse=False):
        is_print = self.is_print
        resblock_num = self.resblock_num
        if is_print:print('build_generator')
        if is_print:print(z.shape)
        x = z
        with tf.variable_scope("generator", reuse=reuse):
            ch = 2
            for i in range(6):
                x = conv(x, channels=ch*2, kernel=4, stride=2, pad=1, sn=self.sn, scope='encode_conv'+str(i))
                # x = batch_norm(x, is_training, scope='batch_norm')
                x = relu(x)
                if is_print:print(x.shape)
                ch = ch * 2
            ch = 1024
            x = deconv(x, channels=ch, kernel=4, stride=1, padding='VALID', use_bias=False, sn=self.sn, scope='deconv')
            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)
            if is_print:print(x.shape)

            for i in range(self.layer_num // 2):
                if self.up_sample:
                    x = up_sample(x, scale_factor=2)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)
                    if is_print:print(x.shape)

                else:
                    x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)
                    if is_print:print(x.shape)

                ch = ch // 2

            # Self Attention
            if self.is_attention:
                x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse)
                if is_print:print('attention!')

            for i in range(self.layer_num // 2, self.layer_num):
                if self.up_sample:
                    x = up_sample(x, scale_factor=2)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)
                    if is_print:print(x.shape)

                else:
                    x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)
                    if is_print:print(x.shape)

                ch = ch // 2

            if self.up_sample:
                x = up_sample(x, scale_factor=2)
                x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, sn=self.sn, scope='G_conv_logit')
                x = tanh(x)

            else:
                x = deconv(x, channels=self.c_dim, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='G_deconv_logit')
                x = tanh(x)

            if is_print:print(x.shape)
            return x

    def generator_attent(self, x, is_training=True, reuse=False):
         is_print = self.is_print
         resblock_num = self.resblock_num
         # kernel_size = self.kernel_size
         kernel_size = 3
         stride = 1
         pad_size = 1
         if is_print:print('build_generator')
         if is_print:print(x.shape)
         with tf.variable_scope("generator", reuse=reuse):
             x = conv(x, channels=1, kernel=kernel_size, stride=stride, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_start')
             ch = 16
             x = conv(x, channels=ch, kernel=kernel_size, stride=stride, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_1')
             if is_print:print(x.shape)
             x = batch_norm(x, is_training, scope='batch_norm_1')
             x = relu(x)

             if self.is_attention:
                if is_print:print('attention!')
                x = self.attention(x, ch, sn=self.sn, scope="attention_g1", reuse=reuse)

             x = conv(x, channels=ch*2, kernel=kernel_size, stride=stride, pad=pad_size, use_bias=False, sn=self.sn, scope='conv_2')
             if is_print:print(x.shape)
             x = batch_norm(x, is_training, scope='batch_norm_2')
             x = relu(x)
             if is_print:print('layer block')
             for i in range(resblock_num):
                 x = conv(x, ch*2, kernel=kernel_size, stride=stride, pad=pad_size,  use_bias=False,sn=False, scope='conv_layer'+str(i))
             if is_print:print(x.shape)
             x = conv(x, channels=ch, kernel=kernel_size, stride=1, pad=pad_size, use_bias=False, sn=self.sn, scope='conv_3')

             # if self.is_attention:
             #    if is_print:print('attention!')
             #    x = self.attention(x, ch, sn=self.sn, scope="attention_g2", reuse=reuse)
             # x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_' + str(i))
             if is_print:print(x.shape)
             x = batch_norm(x, is_training, scope='batch_norm_3')
             x = relu(x)
             # x = self.attention(x, ch, sn=self.sn, scope="attention_2", reuse=reuse)
             x = conv(x, channels=self.c_dim, kernel=kernel_size, stride=1, pad=pad_size, use_bias=False, sn=self.sn, scope='conv_4')
             if is_print:print(x.shape)
             x = tanh(x)
             return x
    def generator_res(self, x, is_training=True, reuse=False):
         is_print = self.is_print
         resblock_num = self.resblock_num
         # kernel_size = self.kernel_size
         kernel_size = 3
         stride = 1
         pad_size = 1
         if is_print:print('build_generator')
         if is_print:print(x.shape)
         with tf.variable_scope("generator", reuse=reuse):
             x = conv(x, channels=1, kernel=kernel_size, stride=stride, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_start')
             # x = self.attention(x, 1, sn=self.sn, scope="attention_g1", reuse=reuse)
             ch = 32
             # x = self.attention(x, 1, sn=self.sn, scope="attention_1", reuse=reuse)
             x = conv(x, channels=ch, kernel=kernel_size, stride=stride, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_1')
             if is_print:print(x.shape)
             x = batch_norm(x, is_training, scope='batch_norm_1')
             x = relu(x)
             if self.is_attention:
                x = self.attention(x, ch, sn=self.sn, scope="attention_g2", reuse=reuse)
                if is_print:print('attention!')
             x = conv(x, channels=ch*2, kernel=kernel_size, stride=stride, pad=pad_size, use_bias=False, sn=self.sn, scope='conv_2')
             if is_print:print(x.shape)
             x = batch_norm(x, is_training, scope='batch_norm_2')
             x = relu(x)
             if is_print:print('residual block')
             for i in range(resblock_num):
                 x = resblock(x, ch*2, use_bias=True,sn=False, scope='resblock'+str(i))
             if is_print:print(x.shape)
             x = conv(x, channels=ch, kernel=kernel_size, stride=1, pad=pad_size, use_bias=False, sn=self.sn, scope='conv_3')
             # x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_' + str(i))
             if is_print:print(x.shape)
             x = batch_norm(x, is_training, scope='batch_norm_3')
             x = relu(x)
             # x = self.attention(x, ch, sn=self.sn, scope="attention_2", reuse=reuse)
             x = conv(x, channels=self.c_dim, kernel=kernel_size, stride=1, pad=pad_size, use_bias=False, sn=self.sn, scope='conv_4')
             if is_print:print(x.shape)
             x = tanh(x)
             return x

    def generator_attention(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        resblock_num = self.resblock_num
        # kernel_size = self.kernel_size
        kernel_size = 3
        pad_size = 1
        if is_print:print('build_generator')
        if is_print:print(x.shape)
        with tf.variable_scope("generator", reuse=reuse):
            ch = 16
            # x = self.attention(x, 1, sn=self.sn, scope="attention_1", reuse=reuse)
            x = conv(x, channels=ch, kernel=4, stride=2, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_1')
            x = batch_norm(x, is_training, scope='batch_norm_1')
            x = relu(x)
            if is_print:print(x.shape)
            x = conv(x, channels=ch*2, kernel=4, stride=2, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_2')
            x = batch_norm(x, is_training, scope='batch_norm_2')
            x = relu(x)
            if is_print:print(x.shape)
            # x = self.attention(x, ch, sn=self.sn, scope="attention_1", reuse=reuse)
            x = conv(x, channels=ch*4, kernel=4, stride=2, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_3')
            x = batch_norm(x, is_training, scope='batch_norm_3')
            if is_print:print(x.shape)
            x = conv(x, channels=ch*4, kernel=kernel_size, stride=1, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_4')
            x = batch_norm(x, is_training, scope='batch_norm_7')
            if is_print:print(x.shape)

            x = self.attention(x, ch*4, sn=self.sn, scope="attention", reuse=reuse)
            if is_print:print('attention !')

            for i in range(resblock_num):
                x = resblock(x, ch*4, use_bias=True,sn=False, scope='resblock'+str(i))
            if is_print:print(x.shape)
            # x = conv(x, channels=ch, kernel=kernel_size, stride=1, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_3')
            x = deconv(x, channels=ch*2 , kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_1')
            if is_print:print(x.shape)
            x = batch_norm(x, is_training, scope='batch_norm_4')
            x = relu(x)
            x = deconv(x, channels=ch*1 , kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_2')
            if is_print:print(x.shape)
            x = batch_norm(x, is_training, scope='batch_norm_5')
            x = relu(x)
            x = deconv(x, channels=ch*1 , kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_3')
            if is_print:print(x.shape)
            x = batch_norm(x, is_training, scope='batch_norm_6')
            x = relu(x)
            # x = self.attention(x, ch, sn=self.sn, scope="attention_2", reuse=reuse)
            x = conv(x, channels=self.c_dim, kernel=kernel_size, stride=1, pad=pad_size, use_bias=True, sn=self.sn, scope='conv_5')
            if is_print:print(x.shape)
            x = tanh(x)
            return x

    def generator_save(self, x, is_training=True, reuse=False):
        layer_num = 4
        is_print = self.is_print
        if is_print:print('build_generator')
        # now, start with sketch 256 * 256 * 1
        if is_print:print(x.shape)
        with tf.variable_scope("generator", reuse=reuse):
            # ch = 1024
            ch = 128
            # x = deconv(x, channels=ch, kernel=4, stride=1, padding='VALID', use_bias=False, sn=self.sn, scope='deconv')
            x = conv(x, channels=ch, kernel=4, stride=2, pad=1, pad_type='zero', use_bias=False, sn=self.sn, scope='conv')
            if is_print:print(x.shape)
            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            if is_print:print('repeat layer : {}'.format(layer_num))
            for i in range(layer_num // 2):
                if self.up_sample:
                    if is_print:print(x.shape)
                    if is_print:print('upsampling!')

                    x = up_sample(x, scale_factor=2)
                    if is_print:print(x.shape)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                else:
                    # x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_' + str(i))
                    x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, pad_type='zero', use_bias=False, sn=self.sn, scope='conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                ch = ch * 2

            if is_print:print(x.shape)
            # Self Attention
            x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse)
            # for i in range(self.resblock_num):
            #     x = self.attention(x, ch, sn=self.sn, scope="attention"+str(i), reuse=reuse)
            #     x = resblock(x, ch, use_bias=True,sn=False, scope='resblock'+str(i))
            if is_print:print('attention !')
            if is_print:print('repeat layer : {}'.format(layer_num))
            for i in range(layer_num // 2, self.layer_num):
                if self.up_sample:
                    x = up_sample(x, scale_factor=2)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                else:
                    x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                ch = ch // 2
            if is_print:print(x.shape)


            if self.up_sample:
                x = up_sample(x, scale_factor=2)
                x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, sn=self.sn, scope='G_conv_logit')
                x = tanh(x)

            else:
                # x = deconv(x, channels=self.c_dim, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='G_deconv_logit')
                x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, pad_type='zero', use_bias=False, sn=self.sn, scope='G_conv_logit')
                x = tanh(x)
            if is_print:print(x.shape)
            # assert False
            return x

    ##################################################################################
    # Discriminator
    ##################################################################################


    def discriminator_my(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        kernel_size = 3
        pad_size = 1
        if is_print:print('build_discriminator')
        if is_print:print(x.shape)
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 64
            if is_print:print(x.shape)
            if self.is_attention:
                x = self.attention(x, 3, sn=self.sn, scope="attention_d", reuse=reuse)
            x = conv(x, channels=ch, kernel=kernel_size, stride=1, pad=1, sn=self.sn, use_bias=True, scope='conv')

            x = lrelu(x, 0.2)
            if is_print:print(x.shape)

            if is_print:print('residual block : {}'.format(self.resblock_num))
            for i in range(4):
                x = resblock(x, ch, use_bias=True,sn=False, scope='resblock'+str(i))
            # Self Attention
            # if is_print:print(x.shape)
            # x = self.attention(x, ch, sn=self.sn, scope="attention_d", reuse=reuse)
            # if is_print:print('attention !')
            if is_print:print(x.shape)
            x = conv(x, channels=1, kernel=kernel_size, stride=1, pad=1, sn=self.sn, use_bias=False, scope='D_logit')
            if is_print:print(x.shape)
            # assert False
            return x

    def discriminator(self, x, is_training=True, reuse=False):
        is_print = self.is_print
        if is_print:print('build_discriminator')
        if is_print:print(x.shape)
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 64
            x = conv(x, channels=ch, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv')
            x = lrelu(x, 0.2)
            if is_print:print(x.shape)

            if is_print:print('repeat layer : {}'.format(self.layer_num))
            for i in range(self.layer_num // 2):
                x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm' + str(i))
                x = lrelu(x, 0.2)

                ch = ch * 2

            # Self Attention
            x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse)
            if is_print:print('attention !')
            if is_print:print(x.shape)

            if is_print:print('repeat layer : {}'.format(self.layer_num))
            # for i in range(self.layer_num // 2, self.layer_num):
            for i in range(12):
                x = resblock(x, ch, use_bias=True,sn=False, scope='resblock'+str(i))
                # x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv_' + str(i))
                # x = batch_norm(x, is_training, scope='batch_norm' + str(i))
                # x = lrelu(x, 0.2)
                # ch = ch * 2
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

    def gradient_penalty(self, real, fake):
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit = self.discriminator(interpolated, reuse=True)

        grad = tf.gradients(logit, interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1)  # l2 norm

        GP = 0

        # WGAN - LP
        if self.gan_type == 'wgan-lp':
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        # images
        # """data load"""
        # self.dataset = tf.data.Dataset.from_tensor_slices((img_l, sketch_l))
        # self.dataset.map(lambda data_img_l, sketch_l: tuple(tf.py_func(read_image,[img_l, sketch_l],[tf.int32, tf.int32])))
        # self.dataset = self.dataset.batch(self.batch_size)
        # self.iterator = self.dataset.make_initializable_iterator()
        # img_stacked, sketch_stacked = self.iterator.get_next()
        # print(img_stacked[0])

#gpu_device = '/gpu:0'
        # noises
        self.z = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.z_dim], name='z')
        self.inputs_photo = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='real_images')
        self.inputs_sketch = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 1], name='sketches')

        """ Loss Function """
        # output of D for real images
        print(self.generator)
        print(self.discriminator)
        print(self.inputs_photo)
        real_logits = self.discriminator(self.inputs_photo)

        # output of D for fake images
        fake_images = self.generator(self.inputs_sketch)
        fake_logits = self.discriminator(fake_images, reuse=True)

        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
            GP = self.gradient_penalty(real=self.inputs, fake=fake_images)
        else :
            GP = 0

        # get loss for discriminator
        self.d_loss = discriminator_loss(self.gan_type, real=real_logits, fake=fake_logits) + GP

        # get loss for generator
        # self.g_loss = generator_loss(self.gan_type, fake=fake_logits)
        self.g_loss = generator_loss(self.gan_type, fake=fake_images)#, real=self.inputs_photo)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        start_d_lr = self.d_learning_rate
        start_g_lr = self.g_learning_rate
        global_step = tf.Variable(0, trainable=False)
        total_learning = self.epoch*self.iteration
        d_lr = tf.train.exponential_decay(start_d_lr, global_step,total_learning,0.99999, staircase=True)
        g_lr = tf.train.exponential_decay(start_g_lr, global_step,total_learning,0.99999, staircase=True)
        self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)
        # self.d_optim = tf.train.AdamOptimizer(d_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
        # self.g_optim = tf.train.AdamOptimizer(g_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)
        #self.d_optim = tf.train.AdagradOptimizer(d_lr).minimize(self.d_loss, var_list=d_vars)
        #self.g_optim = tf.train.AdagradOptimizer(g_lr).minimize(self.g_loss, var_list=g_vars)


        """" Testing """
        # for test
        self.fake_images = self.generator(self.inputs_sketch, is_training=False, reuse=True)

        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)


    def ReadData(self, photo_l, sketch_l):
        photo_num = len(photo_l)
        sketch_num = len(photo_l)
        assert sketch_num == photo_num
        photos = []
        sketches = []
        for i in range(photo_num):
            photo_array, sketch_array = read_image(photo_l[i], sketch_l[i])
            photos.append(photo_array)
            sketches.append(sketch_array)
        print('reading images is done.')
        return photos, sketches

    ##################################################################################
    # Train
    ##################################################################################
    def train(self):
        data_img_l, data_sketch_l = LoadData(self.load_size)
        data_img_l, data_sketch_l = shuffle(data_img_l, data_sketch_l)
        photos, sketches = get_batch(data_img_l, data_sketch_l, self.batch_size, 0)
        # sample_photo = ['/home/sp/Datasets/sketchy/photo/tx_000000000000/airplane/n02691156_58.jpg']
        # sample_sketch = ['/home/sp/Datasets/sketchy/sketch/tx_000000000000/airplane//n02691156_58-2.png']
        self.sample_photo, self.sample_sketch = self.ReadData(photos, sketches)
        #--------------------------------------------------------------------------------------------------
        # dataset = tf.data.Dataset.from_tensor_slices((data_img_l, data_img_l))
        # dataset.map(lambda data_img_l, data_sketch_l: \
        #                 tuple(tf.py_func(read_image_my,[data_img_l, data_sketch_l],\
        #                                  [tf.float32,tf.float32],\
        #                                  )))
        # # dataset.map(resize_image)
        # dataset = dataset.repeat()
        # dataset = dataset.shuffle(buffer_size=200)
        # dataset = dataset.batch(self.batch_size)
        # self.inputs_iterator = dataset.make_initializable_iterator()
        # # self.inputs_iterator = dataset.make_one_shot_iterator()
        # self.data_photo, self.data_sketch = self.inputs_iterator.get_next()
        #---------------------------------------------------
        # dataset_photo = tf.train.string_input_producer(data_img_l)
        # dataset_sketch= tf.train.string_input_producer(data_sketch_l)
        # reader = tf.WholeFileReader()
        # key_photo, value_photo = reader.read(dataset_photo)

        #--------------------------------------------------------------------------------------------------
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        # self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))

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
        past_g_loss = -1.

        for epoch in range(start_epoch, self.epoch):
            data_img_l, data_sketch_l = LoadData(self.load_size)
            data_img_l, data_sketch_l = shuffle(data_img_l, data_sketch_l)
            print('load {} image pair pathes'.format(len(data_img_l)))
            # get batch data
            for idx in range(start_batch_id, self.iteration):
                photos, sketches = get_batch(data_img_l, data_sketch_l, self.batch_size, idx)
                inputs_photos, inputs_sketches = self.ReadData(photos, sketches)
                batch_z = np.random.uniform(-1, 1, [self.batch_size, 1, 1, self.z_dim])
                #---------------------------------------------------
                # self.sess.run(self.inputs_iterator.initializer)
                # batch_photos, batch_sketches = self.sess.run([self.data_photo, self.data_sketch])
                # print(batch_photos[0], batch_photos.shape, type(batch_photos[0]))
                #---------------------------------------------------
                # coord = tf.train.Coordinator()
                # threads = tf.train.start_queue_runners(coord=coord)
                # print(self.sess.run(key_photo))
                # print(self.sess.run(value_photo))
                # coord.request_stop()
                # coord.join(threads)
                # photos = tf.image.decode_jpeg(value_photo)
                # sketches = tf.image.decode_jpeg(value_photo)

                #---------------------------------------------------

                if self.custom_dataset :

                    train_feed_dict = {
                        # self.inputs_photo : tf.cast(batch_photos,dtype=tf.float32),
                        # self.inputs_sketch : tf.cast(batch_sketches, dtype = tf.float32),
                        self.inputs_photo : inputs_photos,
                        self.inputs_sketch : inputs_sketches,
                        self.z: batch_z
                    }

                else :
                    random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
                    # batch_images = self.data[idx*self.batch_size : (idx+1)*self.batch_size]
                    batch_images = self.data[random_index]

                    train_feed_dict = {
                        self.inputs : batch_images,
                        self.z : batch_z
                    }

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # update G network
                g_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss
                print("Epoch: [%2d/%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, self.epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(idx+1, self.print_freq) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.inputs_sketch: self.sample_sketch})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    # save_images(inputs_sketches[:manifold_h * manifold_w, :, :, :],
                    #             [manifold_h, manifold_w],
                    #             './' + self.sample_dir + '/' + self.model_name + '_sketch_{:02d}_{:05d}.png'.format(epoch, idx+1))
                    save_images(samples[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + self.sample_dir + '/' + self.model_name + '_train_{:02d}_{:05d}.png'.format(epoch, idx+1))

                if np.mod(idx+1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}".format(
            self.model_name, self.dataset_name, self.gan_type, self.img_size, self.z_dim, self.sn)

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
