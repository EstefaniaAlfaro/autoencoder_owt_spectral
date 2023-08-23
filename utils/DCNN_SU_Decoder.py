import tensorflow as tf
from tensorflow.keras import regularizers, layers, constraints
from GreaterThanZero import *
from SumToOneConstraint import *
from CustomNormalization import *


class DeepConvolutionalDecoder(tf.keras.layers.Layer):
    def __init__(self, parameters):
        self.parameters = parameters
        super(DeepConvolutionalDecoder, self).__init__()
        self.sum_to_one_constraint = SumToOneConstraints(self.parameters, name='sum_to_one')
        self.custom_normalization = CustomNormalization(self.parameters, name='abundances_normalization')
        self.deep_decoder_su = tf.keras.Sequential([
            layers.Dense(self.parameters['batch_size'] * self.parameters['batch_size'],
                         activation='linear',
                         kernel_constraint=constraints.non_neg(),
                         use_bias=False),
            layers.Reshape((self.parameters['batch_size'], self.parameters['batch_size'])),
    ])

    def call(self, data):
        decoder_ns = self.deep_decoder_su(data)
        return decoder_ns

