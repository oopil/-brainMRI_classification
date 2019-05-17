import tensorflow as tf
from ConvNeuralNet.CNN_ops import *
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

    def CNN_simple(self, x, ch = 32, scope = "CNN", reuse = False):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope(scope, reuse=reuse):
                x = batch_norm(x)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

                ch *= 2
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

                ch *= 2
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

                ch *= 2
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                return x

    def CNN_deep_layer(self, x, scope = "CNN", reuse = False):
        ch = 32
        with tf.variable_scope(scope, reuse=reuse):
            x = batch_norm(x)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

            ch *= 2
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

            ch *= 2
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

            ch *= 2
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
            return x

    def resblock(self, x):
        pass

    def CNN_res(self, x, scope = "CNN", reuse = False):

        with tf.variable_scope(scope, reuse=reuse):
            ch = 32
            with tf.variable_scope(scope, reuse=reuse):
                x = batch_norm(x)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

                ch *= 2
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

                ch *= 2
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.max_pooling3d(inputs=x, pool_size=[2, 2, 2], strides=2)

                ch *= 2
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                x = tf.layers.conv3d(inputs=x, filters=ch, kernel_size=[3, 3, 3], padding='same', activation=self.activ)
                return x

class SimpleNet(Network):
    def model(self, images):
        is_print = False
        # is_print = self.is_print
        if is_print:
            print('build neural network')
            print(images.shape)

        with tf.variable_scope("Model"):
            # images = tf.placeholder(tf.float32, (None, self.ps * 2, self.ps, self.ps, 1), name='inputs')
            lh, rh = tf.split(images, [self.ps, self.ps], 1)
            lh = self.CNN_simple(lh, ch = 32, scope= "Left", reuse=False)
            rh = self.CNN_simple(rh, ch = 32, scope= "Right", reuse=False)
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
            flip_axis = 3
            axis = [False for i in range(5)]
            axis[flip_axis] = True
            rh = tf.reverse(rh, axis=[3])

            # CNN = self.CNN_deep_layer
            CNN = self.CNN_simple

            lh = CNN(lh, ch = 48, scope= "CNN", reuse=False)
            rh = CNN(rh, ch = 48, scope= "CNN", reuse=True)
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
# %%

def cnn_simple_patch(self, x, is_training=True, reuse=False):
    is_print = self.is_print
    if is_print:
        print('build neural network')
        print('input shape : {}'.format(x.shape))

    # x = batch_norm(x)
    with tf.name_scope("preprocess"):
        x = tf.stop_gradient(normalize_tf(x))
        tf.summary.histogram("input_normalize", x)
        lh, rh = tf.stop_gradient(tf.split(x, [self.patch_size, self.patch_size], 1))
    # print(lh.shape, rh.shape)
    # assert False
    with tf.variable_scope("L_cnn", reuse=reuse):
        ch = 64
        lh = self.cnn_layer(lh, ch, ks=4, s=(2, 2, 2), scope='1')
        lh = self.cnn_layer(lh, ch, ks=3, s=(1, 1, 1), scope='2')
        lh = self.cnn_layer(lh, ch, ks=4, s=(2, 2, 2), scope='3')
        lh = self.cnn_layer(lh, ch, ks=3, s=(1, 1, 1), scope='4')
        lh = self.cnn_layer(lh, ch, ks=4, s=(2, 2, 2), scope='5')
        lh = flatten(lh)

    # with tf.variable_scope("R_cnn", reuse=reuse):
    #     ch = 128
    #     rh = self.cnn_layer(rh, ch, ks=4, s=(1, 1, 1), scope='1')
    #     rh = self.cnn_layer(rh, ch, ks=3, s=(1, 1, 1), scope='2') + rh
    #     rh = self.cnn_layer(rh, ch, ks=4, s=(2, 2, 2), scope='3')
    #     rh = self.cnn_layer(rh, ch, ks=3, s=(1, 1, 1), scope='4') + rh
    #     rh = self.cnn_layer(rh, ch, ks=4, s=(2, 2, 2), scope='5')
    #     rh = flatten(rh)
    with tf.variable_scope("fcn", reuse=reuse):
        # x = tf.concat([lh,rh], -1)
        x = lh
        tf.summary.histogram("flatt", x)
        x = self.fc_layer(x, 2048, '1')
        x = self.fc_layer(x, 1024, '2')
        x = self.fc_layer(x, 512, '3')
        x = self.fc_layer(x, 128, '4')
        tf.summary.histogram("flatt_fin", x)
        with tf.variable_scope("last", reuse=reuse):
            x = fully_connected(x, self.class_num, weight_initializer=self.weight_initializer, use_bias=True)
            # x = tf.nn.sigmoid(x)
            # x = tf.nn.softmax(x)
        return x

def cnn_pool(self, x, is_training=True, reuse=False):
    is_print = self.is_print
    if is_print:
        print('build neural network')
        print('input shape : {}'.format(x.shape))

    with tf.name_scope("preprocess"):
        x = tf.stop_gradient(normalize_tf(x))
        # tf.summary.histogram("input_normalize", x)
        # tf.summary.image("input0",x[0])
        # tf.summary.image("input1",x[0])
        # print(x.shape)
        # assert False
        lh, rh = tf.split(x, [self.patch_size, self.patch_size], 1)
        check_lh = tf.expand_dims(lh[:, :, :, 0, 0], -1)
        tf.summary.image("lh",lh[0],max_outputs=5)
        # tf.summary.image("rh",rh[0])

    with tf.variable_scope("L_cnn", reuse=reuse):
        ch = 64
        lh = self.cnn_layer(lh, ch, ks=7, s=(1, 1, 1), scope='1')
        check_lh = tf.expand_dims(lh[:, :, :, 0, 0], -1)
        tf.summary.image("lh1_cnn", check_lh,max_outputs=1)

        lh = self.maxpool(lh, ks=[1,2,2,2,1], s=[1,2,2,2,1], scope='1')
        check_lh = tf.expand_dims(lh[:,:,:,0,0], -1)
        tf.summary.image("lh1_max", check_lh,max_outputs=1)

        lh = self.cnn_layer(lh, ch, ks=5, s=(1, 1, 1), scope='2')
        check_lh = tf.expand_dims(lh[:, :, :, 0, 0], -1)
        tf.summary.image("lh2_cnn", check_lh,max_outputs=1)
        lh = self.maxpool(lh, ks=[1,2,2,2,1], s=[1,2,2,2,1], scope='2')
        # tf.summary.image("lh2", lh[0,:,:,:,1])
        lh = self.cnn_layer(lh, ch, ks=3, s=(1, 1, 1), scope='3')
        check_lh = tf.expand_dims(lh[:, :, :, 0, 0], -1)
        tf.summary.image("lh3_cnn", check_lh,max_outputs=1)
        lh = self.maxpool(lh, ks=[1,2,2,2,1], s=[1,2,2,2,1], scope='3')
        # tf.summary.image("lh3", lh[0,:,:,:,1])

    with tf.variable_scope("fcn", reuse=reuse):
        x = flatten(lh)
        tf.summary.histogram("flatten", x)
        x = self.fc_layer(x, 1024, '1')
        x = self.fc_layer(x, 256, '2')
        x = self.fc_layer(x, 64, '3')
        with tf.variable_scope("last", reuse=reuse):
            x = fully_connected(x, self.class_num, weight_initializer=self.weight_initializer, use_bias=True)
            x = tf.nn.sigmoid(x)
            # x = tf.nn.softmax(x)
        return x
