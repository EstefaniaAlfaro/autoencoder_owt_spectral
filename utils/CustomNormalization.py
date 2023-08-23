import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.python.keras import backend as K


class CustomNormalization(tf.keras.layers.Layer):
    def __init__(self, parameters, **kwargs):
        super(CustomNormalization, self).__init__(**kwargs)
        self.parameters = parameters

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.p = self.add_weight(shape=(input_dim,),
                                 initializer="zeros",
                                 trainable=True)

    def mean_tensor(self, data):
        mean_value = tf.reduce_mean(data, axis=0)
        return mean_value

    def variance_tensor(self, data):
        variance_value = tf.sqrt(tf.math.reduce_variance(data, axis=0) + K.epsilon())
        return variance_value

    def regularization_l2(self, data):
        l2_regularization = regularizers.L2(0.07)(data)
        return l2_regularization

    def call(self, data):
        mean = self.mean_tensor(data)
        sigma = tf.sqrt(self.variance_tensor(data))
        batch_normalization = (data - mean) / sigma + self.p
        return batch_normalization
