from tensorflow.keras import layers, losses, regularizers
from SumToOneConstraint import *
from CustomNormalization import *
from GreaterThanZero import *


class DeepConvolutionalEncoder(tf.keras.Model):
    def __init__(self, parameters):
        self.parameters = parameters
        super(DeepConvolutionalEncoder, self).__init__()
        self.custom_normalization = CustomNormalization(self.parameters, name='abundances_normalization')
        self.deep_encoder_su = tf.keras.Sequential([
            layers.Input(shape=(self.parameters['batch_size'], self.parameters['batch_size'], 1)),
            layers.Conv2D(self.parameters['filter_1'],
                          kernel_size=self.parameters['kernel_1'],
                          strides=self.parameters['stride_1'],
                          padding=self.parameters['padding_1'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Conv2D(self.parameters['filter_2'],
                          kernel_size=self.parameters['kernel_2'],
                          strides=self.parameters['stride_2'],
                          padding=self.parameters['padding_2'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Conv2D(self.parameters['filter_3'],
                          kernel_size=self.parameters['kernel_3'],
                          strides=self.parameters['stride_3'],
                          padding=self.parameters['padding_3'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Conv2D(self.parameters['filter_4'],
                          kernel_size=self.parameters['kernel_4'],
                          strides=self.parameters['stride_4'],
                          padding=self.parameters['padding_4'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Dropout(rate=self.parameters['rate_4']),
            layers.Flatten(),
            layers.Dense(self.parameters['number_end_members'],
                         activation='linear',
                         use_bias=False),
        ])

    def call(self, data):
        architecture = self.deep_encoder_su(data)
        return architecture
