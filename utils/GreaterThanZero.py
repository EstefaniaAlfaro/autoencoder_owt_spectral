import tensorflow as tf
from tensorflow.keras import regularizers


class GreaterThanZero(regularizers.Regularizer):
    def __init__(self, upper_limit):
        super(GreaterThanZero, self).__init__()
        self.upper_limit = upper_limit

    def call(self, data):
        negative_numbers = tf.cast(data < 0, data.type) * data
        boundary = tf.cast(data >= 1.0, data.type) * data
        negative_regularization = -self.upper_limit * tf.reduce_sum(negative_numbers) + \
                                  self.upper_limit * tf.reduce_sum(boundary)
        return negative_regularization
