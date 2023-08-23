from CustomLossFunctions import *
from itertools import product


class MetricsPerformanceSAD:
    def __init__(self, ground_truth, predicted_end_members):
        self.ground_truth = ground_truth
        self.predicted_end_members = predicted_end_members

    def combinations_end_members(self):
        number_end_members = self.ground_truth.shape[-1]
        list_number_abundances = list(range(0, number_end_members))
        product_abundances = list(product(list_number_abundances, list_number_abundances))
        return product_abundances

    def split_end_members(self, spectral_angle_metric):
        number_end_members = self.ground_truth.shape[-1]
        spectral_angle_distance_array = np.asarray(spectral_angle_metric, dtype=object)
        split_chunks_spectral_angle = np.split(spectral_angle_distance_array, number_end_members)
        return split_chunks_spectral_angle

    def best_fit_end_members(self, spectral_angle_metric):
        split_chunk_end_members = self.split_end_members(spectral_angle_metric)
        number_chunks = len(split_chunk_end_members)
        best_spectral_angle_distance = [min(split_chunk_end_members[index])
                                        for index in range(number_chunks)]
        position_best_spectral_angle = [np.argmin(split_chunk_end_members[pos]) for pos in
                                        range(number_chunks)]
        return best_spectral_angle_distance, position_best_spectral_angle

    def order_end_members(self, spectral_angle_metric, spectral_angle_information):
        split_tuple_end_members = self.split_end_members(spectral_angle_information)
        best_spectral_angle_distance, position_best_spectral_angle = self.best_fit_end_members(spectral_angle_metric)
        length_split_tuple = len(split_tuple_end_members)
        ordered_end_members = [split_tuple_end_members[index][position_best_spectral_angle[index]]
                               for index in range(length_split_tuple)]
        return ordered_end_members

    def metrics_sad(self):
        spectral_angle_distance_append = []
        spectral_angle_distance_information = []
        number_combinations_end_members = self.combinations_end_members()
        for index in number_combinations_end_members:
            metrics_spectral_angle = spectral_angle_distance_metric(self.ground_truth[:, index[0]].T,
                                                                    self.predicted_end_members[index[-1], :])
            spectral_angle_distance_append.append(metrics_spectral_angle)
            spectral_angle_distance_information.append((metrics_spectral_angle, index))
        ordered_end_members = self.order_end_members(spectral_angle_distance_append,
                                                     spectral_angle_distance_information)
        return spectral_angle_distance_append, ordered_end_members
