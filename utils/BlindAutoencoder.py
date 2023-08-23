import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses


class BlindAutoencoder(tf.keras.Model):
    def __init__(self, parameters):
        self.parameters = parameters
        super(BlindAutoencoder, self).__init__()
        self.blind_encoder = tf.keras.Sequential([
            layers.Input(shape=(self.parameters['batch_size'], self.parameters['batch_size'], 198)),
            layers.Conv2D(self.parameters['filters_l1'],
                          kernel_size=self.parameters['kernel_size_l1'],
                          strides=self.parameters['strides_l1'],
                          padding=self.parameters['padding_l1'],
                          activation=tf.keras.layers.LeakyReLU(0.02),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                          use_bias=False),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.2),
            layers.Conv2D(self.parameters['filters_l2'],
                          kernel_size=1,
                          strides=parameters['strides_l2'],
                          padding=parameters['padding_l2'],
                          activation=tf.keras.layers.LeakyReLU(0.02),
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                          name='maps_abundances',
                          use_bias=False
                          ),
            layers.BatchNormalization(),
            layers.SpatialDropout2D(0.2),
            layers.Softmax()
        ])
        self.blind_decoder = tf.keras.Sequential([
            layers.Conv2D(parameters['filters_decoder'],
                          kernel_size=parameters['kernel_size_decoder'],
                          strides=parameters['strides_decoder'],
                          padding=parameters['padding_decoder'],
                          activation='linear',
                          kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                          kernel_constraint=tf.keras.constraints.non_neg(),
                          name='end_members',
                          use_bias=False)
        ])

    def call(self, data):
        blind_encoder = self.blind_encoder(data)
        blind_decoder = self.blind_decoder(blind_encoder)
        return blind_decoder

    def get_abundances(self, hyperspectral_data):
        abundances_maps = np.squeeze(self.blind_encoder.predict(np.expand_dims(hyperspectral_data, 0)))
        return abundances_maps

    def get_end_members(self):
        end_members = self.blind_decoder.get_weights()[0]
        if end_members.shape[1] > 1:
            end_members = np.squeeze(end_members).mean(axis=0).mean(axis=0)
        else:
            end_members = np.squeeze(end_members)
        return end_members

