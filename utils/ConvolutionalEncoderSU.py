import tensorflow as tf
from GradientEstimation import *
from SumToOneConstraint import *
from tensorflow.keras import layers, losses


class EncoderSU(tf.keras.Model):
    def __init__(self, parameters):
        super(EncoderSU, self).__init__()
        self.parameters = parameters
        self.hidden_layer_1 = tf.keras.layers.Conv2D(filters=self.parameters['filter_1'],
                                                     kernel_size=self.parameters['size_1'],
                                                     activation=tf.keras.layers.LeakyReLU(0.02),
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                                                     use_bias=False)
        self.hidden_layer_2 = tf.keras.layers.Conv2D(filters=self.parameters['filter_2'],
                                                     kernel_size=self.parameters['size_2'],
                                                     activation=tf.keras.layers.LeakyReLU(0.02),
                                                     strides=1,
                                                     padding='same',
                                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                                                     use_bias=False)
        self.constrain_sum_to_one = SumToOneConstraints(parameters=self.parameters, name='abundances')

    def call(self, data):
        transformation = self.hidden_layer_1(data)
        transformation = tf.keras.layers.BatchNormalization()(transformation)
        transformation = tf.keras.layers.SpatialDropout2D(0.2)(transformation)
        transformation = self.hidden_layer_2(transformation)
        transformation = tf.keras.layers.BatchNormalization()(transformation)
        transformation = tf.keras.layers.SpatialDropout2D(0.2)(transformation)
        transformation = self.constrain_sum_to_one(transformation)
        return transformation
