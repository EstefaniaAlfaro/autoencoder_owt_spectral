import keras.layers
from tensorflow.keras import layers, losses, regularizers
from SumToOneConstraint import *
from CustomNormalization import *
from keras import layers
from keras.layers import Input, concatenate
from keras.models import Model


class NonSymmetricalEncoderSu:
    def __init__(self, parameters):
        self.parameters = parameters
        self.custom_normalization = CustomNormalization(self.parameters, name='abundances_normalization')
        self.sum_to_one = SumToOneConstraints(self.parameters, name='softmax_abundances')

    def non_symmetrical_autoencoder_spectral_un_mixing(self):
        input_network = keras.Input(shape=(self.parameters['batch_size'], self.parameters['batch_size'], 1))
        layer_1 = layers.Conv2D(self.parameters['filter_1'],
                                kernel_size=self.parameters['kernel_1'],
                                strides=self.parameters['stride_1'],
                                padding=self.parameters['padding_1'],
                                activation=tf.keras.layers.LeakyReLU(0.1),
                                kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                                use_bias=False
                                )(input_network)
        layer_2 = layers.Conv2D(self.parameters['filter_2'],
                                kernel_size=self.parameters['kernel_2'],
                                strides=self.parameters['stride_2'],
                                padding=self.parameters['padding_2'],
                                activation=tf.keras.layers.LeakyReLU(0.1),
                                kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                                use_bias=False)(layer_1)
        layer_3 = layers.Conv2D(self.parameters['filter_3'],
                                kernel_size=self.parameters['kernel_3'],
                                strides=self.parameters['stride_3'],
                                padding=self.parameters['padding_3'],
                                activation=tf.keras.layers.LeakyReLU(0.1),
                                kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                                use_bias=False)(layer_2)
        layer_4 = layers.Conv2D(self.parameters['filter_4'],
                                kernel_size=self.parameters['kernel_4'],
                                strides=self.parameters['stride_4'],
                                padding=self.parameters['padding_4'],
                                activation=tf.keras.layers.LeakyReLU(0.1),
                                kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3, seed=123),
                                use_bias=False)(layer_3)
        flatten_layer = layers.Flatten()(layer_4)
        dense_layer = layers.Dense(self.parameters['number_end_members'],
                                   activation='linear', name='abundances_maps')(flatten_layer)
        dense_layer_complete = layers.Dense(self.parameters['batch_size'] * self.parameters['batch_size'],
                                            name='end_members')(dense_layer)
        layers_reshape = layers.Reshape((self.parameters['batch_size'],
                                         self.parameters['batch_size']))(dense_layer_complete)
        abundances_normalization = self.custom_normalization(layers_reshape)
        autoencoder_model = keras.Model(inputs=input_network, outputs=abundances_normalization)
        return autoencoder_model

    def get_abundances_maps(self, data, autoencoder_model):
        feature_abundances_layer = keras.Model(inputs=autoencoder_model.input,
                                               outputs=autoencoder_model.get_layer('abundances_maps').output)
        abundances_maps = feature_abundances_layer.predict(data)
        return abundances_maps

    def get_end_members(self, data, autoencoder_model):
        feature_end_members = keras.Model(inputs=autoencoder_model.input,
                                          outputs=autoencoder_model.get_layer('end_members').output)
        end_members = feature_end_members.get_weights()[0]
        return end_members
