import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses, regularizers
from SumToOneConstraint import *
from GreaterThanZero import *
from CustomNormalization import *


class DeepConvolutionalSpectralUnMixing(tf.keras.Model):
    def __init__(self, parameters):
        self.parameters = parameters
        super(DeepConvolutionalSpectralUnMixing, self).__init__()
        self.sum_to_one_constraint = SumToOneConstraints(self.parameters, name='abundances')
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
            layers.Dropout(rate=parameters['rate_1']),
            layers.Conv2D(self.parameters['filter_2'],
                          kernel_size=self.parameters['kernel_2'],
                          strides=self.parameters['stride_2'],
                          padding=self.parameters['padding_2'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Dropout(rate=parameters['rate_2']),
            layers.Conv2D(self.parameters['filter_3'],
                          kernel_size=self.parameters['kernel_3'],
                          strides=self.parameters['stride_3'],
                          padding=self.parameters['padding_3'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Dropout(rate=parameters['rate_3']),
            layers.Conv2D(self.parameters['filter_4'],
                          kernel_size=self.parameters['kernel_4'],
                          strides=self.parameters['stride_4'],
                          padding=self.parameters['padding_4'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Dropout(rate=parameters['rate_4']),
            layers.Flatten(),
            layers.Dense(9 * self.parameters['number_end_members'], activation=tf.keras.layers.LeakyReLU(0.1)),
            layers.Dense(6 * self.parameters['number_end_members'], activation=tf.keras.layers.LeakyReLU(0.1)),
            layers.Dense(self.parameters['number_end_members'],
                         activation=tf.keras.layers.LeakyReLU(0.1),
                         kernel_constraint=tf.keras.constraints.non_neg()),
            layers.Dropout(rate=parameters['rate_4']),
        ])
        self.deep_decoder_su = tf.keras.Sequential([
            layers.Dense(self.parameters['batch_size'] * self.parameters['batch_size'],
                         activation='linear',
                         kernel_constraint=tf.keras.constraints.non_neg(),
                         kernel_regularizer=regularizers.l2(1e-5)),
            layers.Reshape((self.parameters['batch_size'], self.parameters['batch_size']))
        ])

    def call(self, data):
        deep_encoder_su = self.deep_encoder_su(data)
        batch_normalization = self.custom_normalization(deep_encoder_su)
        deep_decoder_su = self.deep_decoder_su(batch_normalization)
        return deep_decoder_su

    def end_members_extraction(self, data):
        abundances_maps = np.squeeze(self.custom_normalization(self.sum_to_one_constraint(
            self.deep_encoder_su.predict(data))))
        return abundances_maps

    def abundances_maps_extraction(self):
        abundances_maps = self.deep_decoder_su.get_weights()[0]
        return abundances_maps
