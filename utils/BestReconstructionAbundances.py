import numpy as np


def save_dictionary_abundances(best_fit_concatenate, label_abundance):
    best_fit_concatenate_dictionary = {label_abundance[index]: best_fit_concatenate[index] for index in
                                       range(len(best_fit_concatenate))}
    return best_fit_concatenate_dictionary


class BestFitReconstructionAbundances:
    def __init__(self, ground_truth, data_reconstruction, ordered_data):
        self.ground_truth = ground_truth
        self.data_reconstruction = data_reconstruction
        self.ordered_data = ordered_data

    def extract_best_fit_abundances(self):
        ordered_data_length = len(self.ordered_data)
        label_abundance = []
        best_fit_concatenate = []
        for index in range(ordered_data_length):
            data_reconstructed_squeeze = np.squeeze(self.data_reconstruction)
            reshape_ground_truth = np.reshape(self.ground_truth[:, :, self.ordered_data[index][-1][0]], -1)
            reshape_reconstructed_data = np.reshape(data_reconstructed_squeeze[:, :, self.ordered_data[index][-1][-1]],
                                                    -1)
            data_concatenated = np.vstack((reshape_ground_truth, reshape_reconstructed_data)).T
            best_fit_concatenate.append(data_concatenated)
            label_abundance.append('ab_' + 'gt_' + str(self.ordered_data[index][-1][0]) + '_rec_' +
                                   str(self.ordered_data[index][-1][-1]))
        best_fit_concatenate_dictionary = save_dictionary_abundances(best_fit_concatenate, label_abundance)
        return best_fit_concatenate_dictionary


