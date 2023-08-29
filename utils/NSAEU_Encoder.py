from tensorflow.keras import layers, losses, regularizers
from SumToOneConstraint import *
from CustomNormalization import *
from CustomLossFunctions import *


def loss(model, original):
    reconstruction_error = spectral_angle_distance(model(original), original)
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original) + sum(model.losses), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)


class NonSymmetricalConvolutionalEncoder(tf.keras.Model):
    def __init__(self, parameters):
        super(NonSymmetricalConvolutionalEncoder, self).__init__()
        self.parameters = parameters
        self.hidden_layer_1 = tf.keras.layers.Conv2D(self.parameters['filter_1'],
                                                     kernel_size=self.parameters['kernel_1'],
                                                     strides=self.parameters['stride_1'],
                                                     padding=self.parameters['padding_1'],
                                                     activation=tf.keras.layers.LeakyReLU(0.1),
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5,
                                                                                                           seed=123),
                                                     use_bias=False)
        self.hidden_layer_2 = tf.keras.layers.Conv2D(self.parameters['filter_2'],
                                                     kernel_size=self.parameters['kernel_2'],
                                                     strides=self.parameters['stride_2'],
                                                     padding=self.parameters['padding_2'],
                                                     activation=tf.keras.layers.LeakyReLU(0.1),
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5,
                                                                                                           seed=123),
                                                     use_bias=False)
        self.hidden_layer_3 = tf.keras.layers.Conv2D(self.parameters['filter_3'],
                                                     kernel_size=self.parameters['kernel_3'],
                                                     strides=self.parameters['stride_3'],
                                                     padding=self.parameters['padding_3'],
                                                     activation=tf.keras.layers.LeakyReLU(0.1),
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5,
                                                                                                           seed=123),
                                                     use_bias=False)
        self.hidden_layer_4 = tf.keras.layers.Conv2D(self.parameters['number_end_members'],
                                                     kernel_size=1,
                                                     strides=self.parameters['stride_4'],
                                                     padding=self.parameters['padding_4'],
                                                     activation=tf.keras.layers.LeakyReLU(0.1),
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5,
                                                                                                           seed=123),
                                                     use_bias=False)
        self.sum_to_one = SumToOneConstraints(self.parameters, name='abundances')
        self.custom_normalization = CustomNormalization(self.parameters, name='custom_normalization')

    def call(self, data):
        architecture = self.hidden_layer_1(data)
        architecture = self.custom_normalization(architecture)
        architecture = tf.keras.layers.SpatialDropout2D(0.03)(architecture)
        architecture = self.hidden_layer_2(architecture)
        architecture = tf.keras.layers.SpatialDropout2D(0.03)(architecture)
        architecture = self.hidden_layer_3(architecture)
        architecture = tf.keras.layers.SpatialDropout2D(0.03)(architecture)
        architecture = self.hidden_layer_4(architecture)
        architecture = tf.keras.layers.SpatialDropout2D(0.03)(architecture)
        architecture = self.sum_to_one(architecture)
        return architecture
