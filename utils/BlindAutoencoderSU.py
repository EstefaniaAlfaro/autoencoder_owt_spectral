import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses
from SumToOneConstraint import *


class BlindAutoencoderSU(tf.keras.Model):
    def __init__(self, parameters):
        self.parameters = parameters
        super(BlindAutoencoderSU, self).__init__()
        self.sum_to_one = SumToOneConstraints(self.parameters, name='abundances')
        self.blind_encoder_su = tf.keras.Sequential([
            layers.Input(shape=(self.parameters['batch_size'], self.parameters['batch_size'], 1)),
            layers.Conv2D(self.parameters['filter_1'],
                          kernel_size=self.parameters['kernel_1'],
                          strides=self.parameters['stride_1'],
                          padding=self.parameters['padding_1'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.MaxPooling2D(pool_size=(50, 50),
                                strides=(1, 1)),
            layers.Conv2D(self.parameters['filter_2'],
                          kernel_size=self.parameters['kernel_2'],
                          strides=self.parameters['stride_2'],
                          padding=self.parameters['padding_2'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.MaxPooling2D(pool_size=(25, 25),
                                strides=(1, 1)),
            layers.Conv2D(self.parameters['filter_3'],
                          kernel_size=self.parameters['kernel_3'],
                          strides=self.parameters['stride_3'],
                          padding=self.parameters['padding_3'],
                          activation=tf.keras.layers.LeakyReLU(0.1),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                          use_bias=False),
            layers.Flatten(),
            layers.Dense(9 * 4, activation=tf.keras.layers.LeakyReLU(0.1)),
            layers.Dense(6 * 4, activation=tf.keras.layers.LeakyReLU(0.1)),
            layers.Dropout(rate=0.01),
            layers.Dense(3 * 4, activation='linear',
                         kernel_constraint=tf.keras.constraints.non_neg()),
            layers.Dense(4, activation=tf.keras.layers.LeakyReLU(0.1))
        ])
        self.blind_decoder_su = tf.keras.Sequential([
            layers.Dense(self.parameters['batch_size'] * self.parameters['batch_size'],
                         activation='linear',
                         kernel_constraint=tf.keras.constraints.non_neg()),
            layers.Reshape((self.parameters['batch_size'], self.parameters['batch_size']))
        ])

    def call(self, data):
        encoder_su = self.blind_encoder_su(data)
        decoder_su = self.blind_decoder_su(encoder_su)
        return decoder_su

    def get_end_members(self, data):
        abundances_maps = np.squeeze(self.blind_encoder_su.predict(data))
        return abundances_maps

    def get_abundances_maps(self):
        end_members = self.blind_decoder_su.get_weights()[0]
        return end_members

