import tensorflow as tf
from keras.layers.convolutional import UpSampling3D
# import keras.layers.convolutional as UpSampling3D
from ConvNeuralNet.CNN_ops import *
##################################################################################
# Custom Operation
##################################################################################
class Network:
    def __init__(self,
                 weight_initializer,
                 activation,
                 class_num,
                 patch_size,
                 patch_num,
                 ):
        self.weight_init = weight_initializer
        self.activ = activation
        self.cn = class_num
        self.ps = patch_size
        self.pn = patch_num

    def conv_3d(self, x, ch, ks, padding, activ, st = (1,1,1)):
        return tf.layers.conv3d(inputs=x, filters=ch, kernel_size=ks, padding=padding, activation=activ, strides=st)

    def deconv_3d(self, x, ch, ks, padding, activ, st = (1,1,1)):
        return tf.layers.conv3d_transpose(inputs=x, filters=ch, kernel_size=ks, padding=padding, activation=activ, strides=st)

    def maxpool_3d(self, x, ps, st):
        return tf.layers.max_pooling3d(inputs=x, pool_size=ps, strides=st)

    def CNN_simple(self, x, ch = 32, scope = "CNN", reuse = False):
        k3 = [3,3,3]
        k4 = [4,4,4]
        k5 = [5,5,5]
        k7 = [7,7,7]
        # kernel_pool = [2,2,2]
        kernel_pool = [3,3,3]
        with tf.variable_scope(scope, reuse=reuse):
            x = batch_norm(x)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)

            ch *= 2
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)

            ch *= 2
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)

            ch *= 2
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.conv_3d(x, ch, k5, 'same', self.activ)
            x = self.maxpool_3d(x, kernel_pool, st=2)
            return x

    def resblock(self, input, ch_in, ch_out, ks):
        x = self.conv_3d(input, ch_out, 1, 'same', self.activ)
        x = self.conv_3d(x, ch_out, ks, 'same', self.activ)
        if ch_in != ch_out:
            input = self.conv_3d(input, ch_out, ks, 'same', self.activ)
        return self.conv_3d(x, ch_out, ks, 'same', self.activ) + input

    def CNN_nopool(self, x, ch = 32, scope = "CNN", reuse = False):
        with tf.variable_scope(scope, reuse=reuse):
            x = batch_norm(x)
            kernel = [4,4,4]
            x = self.conv_3d(x, ch, kernel, 'same', self.activ)
            x = self.conv_3d(x, ch, kernel, 'same', self.activ, st=(2,2,2))

            ch *= 2
            x = self.conv_3d(x, ch, kernel, 'same', self.activ)
            x = self.conv_3d(x, ch, kernel, 'same', self.activ, st=(2,2,2))

            ch *= 2
            x = self.conv_3d(x, ch, kernel, 'same', self.activ)
            x = self.conv_3d(x, ch, kernel, 'same', self.activ, st=(2,2,2))

            ch *= 2
            x = self.conv_3d(x, ch, kernel, 'same', self.activ)
            x = self.conv_3d(x, ch, kernel, 'same', self.activ, st=(2,2,2))
            return x

    def CNN_deep_layer(self, x, scope = "CNN", reuse = False):
        ch = 32
        with tf.variable_scope(scope, reuse=reuse):
            x = batch_norm(x)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            return x

    def attention(self, x):
        pass

    def res_attention(self, x):
        pass

    def CNN_res(self, x, ch, scope = "CNN", reuse = False):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope(scope, reuse=reuse):
                x = batch_norm(x)
                x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
                x = self.resblock(x,ch,[3, 3, 3])
                x = self.maxpool_3d(x, [2, 2, 2], st=2)

                ch *= 2
                x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
                x = self.resblock(x, ch, [3, 3, 3])
                x = self.maxpool_3d(x, [2, 2, 2], st=2)

                ch *= 2
                x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
                x = self.resblock(x, ch, [3, 3, 3])
                x = self.maxpool_3d(x, [2, 2, 2], st=2)

                # ch *= 2
                # x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
                # x = self.resblock(x, ch, [3, 3, 3])
                return x

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

def cnn_layer(self, x, ch, ks, s, scope):
    with tf.name_scope(scope):
        # return lrelu(conv3d(x, ch, ks=ks, s=s, stddev=self.weight_stddev, name=scope))
        return relu(conv3d(x, ch, ks=ks, s=s, stddev=self.weight_stddev, name=scope), scope=scope)

def maxpool(self, x, ks, s, scope):
    with tf.name_scope(scope):
        return max_pooling(x, ks=ks, s=s)

##################################################################################
# Convolutional Neural Network Model
##################################################################################
class Simple(Network):
    def model(self, images):
        is_print = False
        if is_print:
            print('build neural network')
            print(images.shape)

        channel = 32
        CNN = self.CNN_simple
        split_form = [self.ps for _ in range(self.pn)]
        with tf.variable_scope("Model"):
            # lh, rh = tf.split(images, split_form, 1)
            split_array = tf.split(images, split_form, 1)
            cnn_features = []
            for i, array in enumerate(split_array):
                array = CNN(array, ch=channel, scope="CNN"+str(i), reuse=False)
                array = tf.layers.flatten(array)
                cnn_features.append(array)
            # CNN = self.CNN_deep_layer

            # channel = 40
            # lh = CNN(lh, ch = channel, scope= "LCNN", reuse=False)
            # rh = CNN(rh, ch = channel, scope= "RCNN", reuse=False)
            with tf.variable_scope("FCN"):
                # lh = tf.layers.flatten(lh)
                # rh = tf.layers.flatten(rh)
                # x = tf.concat([lh, rh], -1)
                x = tf.concat(cnn_features, -1)
                x = tf.layers.dense(x, units=1024, activation=self.activ)
                x = tf.layers.dense(x, units=512, activation=self.activ)
                x = tf.layers.dense(x, units=self.cn, activation=tf.nn.softmax)
                # x = tf.layers.dense(x, units=self.cn, activation=tf.nn.sigmoid)
                y = x
        return y

class Siamese(Network):
    def model(self, images):
        is_print = False
        # is_print = self.is_print
        if is_print:
            print('build neural network')
            print(images.shape)

        with tf.variable_scope("Model"):
            # images = tf.placeholder(tf.float32, (None, self.ps * 2, self.ps, self.ps, 1), name='inputs')
            lh, rh = tf.split(images, [self.ps, self.ps], 1)
            # output_num = 3
            # tf.summary.image('lh_orig_1',lh[0],max_outputs=output_num)
            # tf.summary.image('lh_orig_2',lh[0],max_outputs=output_num)
            # tf.summary.image('rh_orig',rh[0],max_outputs=output_num)
            flip_axis = 3 # 3
            # axis = [False for i in range(5)]
            # axis[flip_axis] = True
            rh = tf.reverse(rh, axis=[flip_axis])
            # tf.summary.image('rh',rh[0],max_outputs=output_num)
            # CNN = self.CNN_deep_layer
            CNN = self.CNN_simple

            channel = 32
            # channel = 40
            lh = CNN(lh, ch = channel, scope= "CNN", reuse=False)
            rh = CNN(rh, ch = channel, scope= "CNN", reuse=True)

            with tf.variable_scope("FCN"):
                lh = tf.layers.flatten(lh)
                rh = tf.layers.flatten(rh)
                x = tf.concat([lh, rh], -1)
                # x = tf.subtract(lh,rh)
                x = tf.layers.dense(x, units=2048, activation=self.activ)
                x = tf.layers.dense(x, units=512, activation=self.activ)
                x = tf.layers.dense(x, units=self.cn, activation=tf.nn.softmax)
                # x = tf.layers.dense(x, units=self.cn, activation=tf.nn.sigmoid)
                y = x
        return y
# %%
class Residual(Network):
    def model(self, images):
        is_print = False
        # is_print = self.is_print
        if is_print:
            print('build neural network')
            print(images.shape)

        with tf.variable_scope("Model"):
            # images = tf.placeholder(tf.float32, (None, self.ps * 2, self.ps, self.ps, 1), name='inputs')
            lh, rh = tf.split(images, [self.ps, self.ps], 1)
            # CNN = self.CNN_deep_layer
            # CNN = self.CNN_simple
            CNN = self.CNN_res

            channel = 32
            lh = CNN(lh, ch = channel, scope= "CNN", reuse=False)
            rh = CNN(rh, ch = channel, scope= "CNN", reuse=True)
            with tf.variable_scope("FCN"):
                lh = tf.layers.flatten(lh)
                rh = tf.layers.flatten(rh)
                x = tf.concat([lh, rh], -1)
                x = tf.layers.dense(x, units=2048, activation=self.activ)
                x = tf.layers.dense(x, units=512, activation=self.activ)
                x = tf.layers.dense(x, units=self.cn, activation=tf.nn.softmax)
                # x = tf.layers.dense(x, units=self.cn, activation=tf.nn.sigmoid)
                y = x
        return y

class Attention(Network):
    def CNN_attention(self, input, ch = 16, scope = "resAttention", reuse = False):
        t = 1
        p = 1
        r = 1
        with tf.variable_scope(scope):
            with tf.variable_scope("first_residual_blocks"):
                for i in range(p):
                    input = self.conv_3d(input, ch, 3, 'same', self.activ)
                    # input = self.resblock(input, 1, ch, ks=3)

            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(t):
                    output_trunk = self.conv_3d(output_trunk, ch, 3, 'same', self.activ)

            with tf.variable_scope("soft_mask_branch"):
                with tf.variable_scope("down_sampling_1"):
                    output_soft_mask = self.maxpool_3d(input, ps=2, st=2)
                    for i in range(r):
                        output_soft_mask = self.conv_3d(output_soft_mask, ch, 3,'same',  self.activ)
                        # output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("skip_connection"):
                    output_skip_connection = self.conv_3d(input, ch, 3, 'same', self.activ)

                with tf.variable_scope("up_sampling_1"):
                    for i in range(r):
                        output_soft_mask = self.conv_3d(output_soft_mask, ch, 3, 'same', self.activ)
                        # output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)
                    output_soft_mask = self.deconv_3d(output_soft_mask, ch, 3, 'same', self.activ, st=2)

                # add skip connection
                output_soft_mask += output_skip_connection

                with tf.variable_scope("output"):
                    output_soft_mask = self.conv_3d(output_soft_mask, ch, 3, 'same', self.activ)
                    # output_soft_mask = self.conv_3d(output_soft_mask, ch, k3, 'same', self.activ)
                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(t):
                    output = self.conv_3d(output, ch, 3,'same',  self.activ)
                    # output = self.resblock(output, ch, ch, ks=3)
            return output

    def CNN_attention_res(self, input, ch = 16, scope = "resAttention", reuse = False):
        k3 = [3, 3, 3]
        k4 = [4, 4, 4]
        k5 = [5, 5, 5]
        k7 = [7, 7, 7]
        t = 1
        p = 1
        r = 1
        with tf.variable_scope(scope):
            # residual blocks(TODO: change this function)
            with tf.variable_scope("first_residual_blocks"):
                for i in range(p):
                    input = self.resblock(input, 1, ch, ks=3)

            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(t):
                    output_trunk = self.resblock(output_trunk, ch, ch, ks=3)

            with tf.variable_scope("soft_mask_branch"):
                with tf.variable_scope("down_sampling_1"):
                    output_soft_mask = self.maxpool_3d(input, ps=2, st=2)
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("skip_connection"):
                    output_skip_connection = self.resblock(input, ch, ch, ks=3)

                with tf.variable_scope("up_sampling_1"):
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)
                    output_soft_mask = self.deconv_3d(output_soft_mask, ch, k3, 'same', self.activ, st=2)

                # add skip connection
                output_soft_mask += output_skip_connection

                with tf.variable_scope("output"):
                    output_soft_mask = self.conv_3d(output_soft_mask, ch, k3, 'same', self.activ)
                    # output_soft_mask = self.conv_3d(output_soft_mask, ch, k3, 'same', self.activ)
                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(t):
                    output = self.resblock(output_soft_mask, ch, ch, ks=3)

            return output

    def CNN_attention_default(self, input, ch = 16, scope = "resAttention", reuse = False):
        k3 = [3, 3, 3]
        k4 = [4, 4, 4]
        k5 = [5, 5, 5]
        k7 = [7, 7, 7]
        t = 1
        p = 1
        r = 1
        with tf.variable_scope(scope):
            # residual blocks(TODO: change this function)
            with tf.variable_scope("first_residual_blocks"):
                for i in range(t):
                    x = self.resblock(input, 1, ch, ks=3)

            with tf.variable_scope("trunk_branch"):
                output_trunk = x
                for i in range(t):
                    output_trunk = self.resblock(x, ch, ch, ks=3)

            with tf.variable_scope("soft_mask_branch"):
                with tf.variable_scope("down_sampling_1"):
                    # max pooling
                    filter_ = 3
                    output_soft_mask = self.maxpool_3d(input, ps=2, st=2)
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("skip_connection"):
                    output_skip_connection = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("down_sampling_2"):
                    # max pooling
                    filter_pool = [2,2,2]
                    output_soft_mask = self.maxpool_3d(output_soft_mask, ps=2, st=2)
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                with tf.variable_scope("up_sampling_1"):
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)

                    # interpolation
                    output_soft_mask = self.deconv_3d(output_soft_mask, ch, k3, 'same', self.activ, st=2)

                    # output_soft_mask = UpSampling3D(size=(2,2,2))(output_soft_mask)
                    # output_soft_mask = UpSampling2D([2, 2])(output_soft_mask)

                # add skip connection
                output_soft_mask += output_skip_connection

                with tf.variable_scope("up_sampling_2"):
                    for i in range(r):
                        output_soft_mask = self.resblock(output_soft_mask, ch, ch, ks=3)
                    # interpolation
                    output_soft_mask = self.deconv_3d(output_soft_mask, ch, k3, 'same', self.activ, st=2)
                    # output_soft_mask = UpSampling3D(size=(2,2,2))(output_soft_mask)

                with tf.variable_scope("output"):
                    output_soft_mask = self.conv_3d(output_soft_mask, ch, k3, 'same', self.activ)
                    output_soft_mask = self.conv_3d(output_soft_mask, ch, k3, 'same', self.activ)
                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(t):
                    output = self.resblock(output_soft_mask, ch, ch, ks=3)

            return output

    def model(self, images):
        is_print = False
        if is_print:
            print('build neural network')
            print(images.shape)

        channel = 32
        CNN = self.CNN_attention_res

        split_form = [self.ps for _ in range(self.pn)]
        with tf.variable_scope("Model"):
            # lh, rh = tf.split(images, split_form, 1)
            split_array = tf.split(images, split_form, 1)
            cnn_features = []
            for i, array in enumerate(split_array):
                array = CNN(array, ch=channel, scope="CNN"+str(i), reuse=False)
                array = self.maxpool_3d(array, ps=2, st=2)
                array = CNN(array, ch=channel*2, scope="CNN2"+str(i), reuse=False)
                array = self.maxpool_3d(array, ps=2, st=2)
                # array = CNN(array, ch=channel*4, scope="CNN3"+str(i), reuse=False)
                # array = self.maxpool_3d(array, ps=2, st=2)

                # we need to reduce the image size..
                # if i use residual attention module only one block, i can't reduce the image
                # for i in range(2):
                #     array = self.resblock(array, channel, channel*2, ks=3)
                #     # array = self.conv_3d(array, channel, 3, 'same', self.activ)
                #     array = self.maxpool_3d(array, ps=2, st=2)

                print(np.shape(array))
                array = tf.layers.flatten(array)
                cnn_features.append(array)
            # CNN = self.CNN_deep_layer

            # channel = 40
            # lh = CNN(lh, ch = channel, scope= "LCNN", reuse=False)
            # rh = CNN(rh, ch = channel, scope= "RCNN", reuse=False)
            # print(np.shape(cnn_features))
            with tf.variable_scope("FCN"):
                # lh = tf.layers.flatten(lh)
                # rh = tf.layers.flatten(rh)
                # x = tf.concat([lh, rh], -1)
                x = tf.concat(cnn_features, -1)
                x = tf.layers.dense(x, units=1024, activation=self.activ)
                x = tf.layers.dense(x, units=512, activation=self.activ)
                # x = tf.layers.dense(x, units=self.cn, activation=tf.nn.softmax)
                x = tf.layers.dense(x, units=self.cn, activation=tf.nn.sigmoid)
                y = x
        return y