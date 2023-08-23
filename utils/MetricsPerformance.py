import tensorflow as tf
from itertools import product
import numpy as np


class MetricsPerformance:
    def __init__(self, ground_truth, prediction):
        self.ground_truth = ground_truth
        self.prediction = prediction

    def combinations_abundances_maps(self):
        number_abundances_maps = self.ground_truth.shape[-1]
        list_number_abundances = list(range(0, number_abundances_maps))
        product_abundances = list(product(list_number_abundances, list_number_abundances))
        return product_abundances

    def split_abundances_maps(self, root_mean_square_append):
        number_abundances_maps = self.ground_truth.shape[-1]
        root_mean_square_array = np.asarray(root_mean_square_append, dtype=object)
        split_chunks_abundances = np.split(root_mean_square_array, number_abundances_maps)
        return split_chunks_abundances

    def best_fit_abundances_maps(self, root_mean_square_append):
        split_chunks_abundances = self.split_abundances_maps(root_mean_square_append)
        number_chunks = len(split_chunks_abundances)
        best_root_mean_square_error = [min(split_chunks_abundances[index])
                                                                 for index in range(number_chunks)]
        position_best_error = [np.argmin(split_chunks_abundances[pos]) for pos in range(number_chunks)]
        return best_root_mean_square_error, position_best_error

    def order_abundances_maps(self, root_mean_square_append, root_mean_square_information):
        split_tuple_match_abundances = self.split_abundances_maps(root_mean_square_information)
        best_root_mean_square_error, position_best_error = self.best_fit_abundances_maps(root_mean_square_append)
        length_split_tuple = len(split_tuple_match_abundances)
        ordered_abundances_maps = [split_tuple_match_abundances[index][position_best_error[index]]
                                   for index in range(length_split_tuple)]
        return ordered_abundances_maps

    def metric_root_mean_square_error(self):
        root_mean_square_append = []
        root_mean_square_information = []
        product_abundances = self.combinations_abundances_maps()
        for index in product_abundances:
            squeeze_predicted = tf.squeeze(self.prediction)
            root_mean_square_error = tf.keras.metrics.RootMeanSquaredError()
            root_mean_square_error.update_state(self.ground_truth[:, :, index[0]], squeeze_predicted[:, :, index[-1]])
            metric_root_mean_square_error = root_mean_square_error.result().numpy()
            root_mean_square_append.append(metric_root_mean_square_error)
            root_mean_square_information.append((metric_root_mean_square_error, index))
        ordered_abundances_maps = self.order_abundances_maps(root_mean_square_append, root_mean_square_information)
        return root_mean_square_append, ordered_abundances_maps
