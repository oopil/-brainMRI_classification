import tensorflow as tf
import NeuralNet.NN_ops

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
class SimplestNet:
    def __init__(self, weight_initializer, activation):
        pass

    def model(self, x):
        layer_num = 3
        # is_print = self.is_print
        # if is_print:
        #     print('build neural network')
        #     print(x.shape)
        with tf.variable_scope("neuralnet", reuse=reuse):
            x = self.fc_layer(x, 512, 'fc_input_1')
            x = self.fc_layer(x, 1024, 'fc_input_2')
            for i in range(layer_num):
                x = self.fc_layer(x, 1024, 'fc' + str(i))
            x = self.fc_layer(x, 512, 'fc_1')
            x = self.fc_layer(x, 256, 'fc_fin')
            # x = self.fc_layer(x, self.class_num, 'fc_last')
            x = fully_connected(x, self.class_num, \
                                weight_initializer=self.weight_initializer, use_bias=True, scope='fc_last')
            # tf.summary.histogram('last_active', x)
            return x

        pass

class SimpleNet:
    def __init__(self, weight_initializer, activation, class_num):
        self.weight_init = weight_initializer
        self.activ = activation
        self.class_num = class_num
        pass

    def model(self, x):
        layer_num = 3
        is_print = False
        # is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet"):
            x = tf.layers.dense(x, units=512, activation=self.activ)
            x = tf.layers.dense(x, units=1024, activation=self.activ)
            for i in range(layer_num):
                x = tf.layers.dense(x, units=1024, activation=self.activ)
            x = tf.layers.dense(x, units=512, activation=self.activ)
            x = tf.layers.dense(x, units=256, activation=self.activ)
            x = tf.layers.dense(x, units=self.class_num, activation=tf.nn.softmax)
        return x

class ResNet:
    def __init__(self, weight_initializer, activation, class_num):
        self.weight_init = weight_initializer
        self.activ = activation
        self.class_num = class_num
        pass

    def model(self, x):
        def resblock(x, ch):
            x1 = tf.layers.dense(x, units=ch, activation=self.activ)
            return tf.layers.dense(x, units=ch, activation=self.activ) + x

        layer_num = 3
        is_print = False
        # is_print = self.is_print
        if is_print:
            print('build neural network')
            print(x.shape)
        with tf.variable_scope("neuralnet"):
            x = tf.layers.dense(x, units=512, activation=self.activ)
            # x = resblock(x, 512)
            x = tf.layers.dense(x, units=1024, activation=self.activ)
            for i in range(layer_num):
                x = resblock(x, 1024)
            x = tf.layers.dense(x, units=512, activation=self.activ)
            x = tf.layers.dense(x, units=256, activation=self.activ)
            x = tf.layers.dense(x, units=self.class_num, activation=tf.nn.softmax)
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
