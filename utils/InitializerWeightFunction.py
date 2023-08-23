import tensorflow as tf


class InitializerWeights(tf.keras.initializers.Initializer):
    def __init__(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(shape, mean=self.mean, stddev=self.standard_deviation)

    def get_config(self):
        return {'mean': self.mean, 'standard_deviation': self.standard_deviation}
