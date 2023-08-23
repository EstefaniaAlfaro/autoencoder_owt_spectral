import keras
from tensorflow.keras import layers, losses
from InitializerWeightFunction import *
from CustomLossFunctions import *
from GradientEstimation import *
import numpy as np


def get_end_members(get_end_members_weights):
    end_members = np.squeeze(get_end_members_weights).mean(axis=0).mean(axis=0)
    return end_members


def model_autoencoder_reduce(parameters, mean, standard_deviation):
    input_data = keras.Input(shape=(parameters['batch_size'], parameters['batch_size'], 198))
    hidden_layer_1 = layers.Conv2D(parameters['filters_l1'],
                                   kernel_size=parameters['kernel_size_l1'],
                                   strides=parameters['strides_l1'],
                                   padding=parameters['padding_l1'],
                                   activation=tf.keras.layers.LeakyReLU(0.02),
                                   kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                                   use_bias=False)(input_data)
    batch_normalization = tf.keras.layers.BatchNormalization()(hidden_layer_1)
    spatial_dropout = tf.keras.layers.SpatialDropout2D(0.2)(batch_normalization)
    encoder = layers.Conv2D(parameters['filters_l2'],
                            kernel_size=1,
                            strides=parameters['strides_l2'],
                            padding=parameters['padding_l2'],
                            activation=tf.keras.layers.LeakyReLU(0.02),
                            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                            name='maps_abundances',
                            use_bias=False)(spatial_dropout)
    batch_normalization_intermediate = tf.keras.layers.BatchNormalization()(encoder)
    spatial_dropout_intermediate = tf.keras.layers.SpatialDropout2D(0.2)(batch_normalization_intermediate)
    sum_to_one_force_function = tf.nn.softmax(spatial_dropout_intermediate * 5, name='softmax_abundances')
    decoder = layers.Conv2D(parameters['filters_decoder'],
                            kernel_size=parameters['kernel_size_decoder'],
                            strides=parameters['strides_decoder'],
                            padding=parameters['padding_decoder'],
                            activation='linear',
                            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                            kernel_constraint=tf.keras.constraints.NonNeg(),
                            name='end_members',
                            use_bias=False)(sum_to_one_force_function)
    autoencoder = keras.Model(input_data, decoder)
    autoencoder_custom = CustomGradientModel(input_data, decoder)
    return autoencoder_custom
