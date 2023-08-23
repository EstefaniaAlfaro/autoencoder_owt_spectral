import tensorflow as tf
from tensorflow.keras import regularizers


def orthogonal_sparse_prior(term_1, term_2):
    dots = 0.0
    term_1 = tf.linalg.l2_normalize(term_1, axis=0)
    sum_contribution = []
    for index_row in range(term_2):
        for index_col in range(index_row + 1, term_2):
            orthogonal_sparse_1 = term_1[:, index_row]
            orthogonal_sparse_2 = term_1[:, index_col]
            dot_product = tf.reduce_sum(orthogonal_sparse_1 * orthogonal_sparse_2, axis=0)
            sum_contribution = dots + dot_product
    return sum_contribution


class SumToOneConstraints(tf.keras.layers.Layer):
    def __init__(self, parameters, **kwargs):
        super(SumToOneConstraints, self).__init__(**kwargs)
        self.number_outputs = parameters['number_end_members']
        self.parameters = parameters

    def build(self, input_data):
        assert len(input_data) >= 2
        input_length = input_data[-1]

    def regularization_l1(self, data):
        l1_regularization = regularizers.L1(1.0)(data)
        return l1_regularization

    def regularization_l2(self, data):
        l2_regularization = regularizers.L2(0e-8)(data)
        return l2_regularization

    def orthogonal_sparse_regularization(self, data):
        orthogonal_sparse_results = (orthogonal_sparse_prior(data, self.number_outputs)) * 0.5
        return orthogonal_sparse_results

    def call(self, data):
        data = tf.nn.softmax(self.parameters['scale'] * data)
        return data
