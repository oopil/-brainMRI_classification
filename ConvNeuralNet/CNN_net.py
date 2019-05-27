import tensorflow as tf
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

    def conv_3d(self, x, ch, ks, padding, activ):
        return tf.layers.conv3d(inputs=x, filters=ch, kernel_size=ks, padding=padding, activation=activ)

    def maxpool_3d(self, x, ps, st):
        return tf.layers.max_pooling3d(inputs=x, pool_size=ps, strides=st)

    def CNN_simple(self, x, ch = 32, scope = "CNN", reuse = False):
        with tf.variable_scope(scope, reuse=reuse):
            x = batch_norm(x)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.maxpool_3d(x, [2, 2, 2], st=2)

            ch *= 2
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
            x = self.conv_3d(x, ch, [3, 3, 3], 'same', self.activ)
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

    def resblock(self, x, ch, ks):
        i = self.conv_3d(x, ch, ks, 'same', self.activ)
        return self.conv_3d(i, ch, ks, 'same', self.activ) + x

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

        with tf.variable_scope("Model"):
            lh, rh = tf.split(images, [self.ps, self.ps], 1)
            # CNN = self.CNN_deep_layer
            CNN = self.CNN_simple

            # channel = 32
            channel = 48
            lh = CNN(lh, ch = channel, scope= "LCNN", reuse=False)
            rh = CNN(rh, ch = channel, scope= "RCNN", reuse=False)
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

            # channel = 32
            channel = 48
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
            CNN = self.CNN_simple

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