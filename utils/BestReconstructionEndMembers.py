import numpy as np


def save_dictionary_end_members(best_fit_concatenate, label_end_member):
    best_fit_end_member_dictionary = {label_end_member[index]: best_fit_concatenate[index] for index in
                                       range(len(best_fit_concatenate))}
    return best_fit_end_member_dictionary


class BestFitReconstructionEndMembers:
    def __init__(self, ground_truth, data_reconstruction, ordered_data):
        self.ground_truth = ground_truth
        self.data_reconstruction = data_reconstruction
        self.ordered_data = ordered_data

    def extract_best_fit_end_member(self):
        ordered_length = len(self.ordered_data)
        label_end_members = []
        best_fit_end_member = []
        ground_truth = self.ground_truth.T
        for index in range(ordered_length):
            end_member_reconstructed_squeeze = np.squeeze(self.data_reconstruction)
            best_fit_ground_truth = ground_truth[self.ordered_data[index][-1][0]]
            best_end_member_reconstructed = end_member_reconstructed_squeeze[self.ordered_data[index][-1][-1]]
            data_concatenate_end_member = np.vstack((best_fit_ground_truth, best_end_member_reconstructed)).T
            label_end_members.append('endM_' + 'gt_' + str(self.ordered_data[index][-1][0]) + '_rec_' +
                                     str(self.ordered_data[index][-1][-1]))
            best_fit_end_member.append(data_concatenate_end_member)
        best_fit_end_member_dictionary = save_dictionary_end_members(best_fit_end_member, label_end_members)
        return best_fit_end_member_dictionary