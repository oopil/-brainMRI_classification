##################################################################################
# Neural Network Model
##################################################################################
import tensorflow as tf
import NeuralNet.NN_ops
class SimpleNet:
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

# class NeuralNetSimple:
#     def __init__(self, weight_initializer, activation, class_num):
#         self.weight_init = weight_initializer
#         self.activ = activation
#         self.class_num = class_num
#         pass

def NeralNetSimple(self, x, weight_initializer, activation, class_num):
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

            #
            # x = self.fc_layer(x, 512, 'fc_input_1')
            # x = self.fc_layer(x, 1024, 'fc_input_2')
            # for i in range(layer_num):
            #     x = self.fc_layer(x, 1024, 'fc' + str(i))
            # x = self.fc_layer(x, 512, 'fc_1')
            # x = self.fc_layer(x, 256, 'fc_fin')
            # # x = self.fc_layer(x, self.class_num, 'fc_last')
            # x = fully_connected(x, self.class_num, \
            #                     weight_initializer=self.weight_init, use_bias=True, scope='fc_last')
            # # tf.summary.histogram('last_active', x)
            # return x


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
