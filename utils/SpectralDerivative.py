import numpy as np
import math


def hypercube_representation(first_circulant_shifted):
    concatenate_hypercube = np.dstack((first_circulant_shifted[0], first_circulant_shifted[1]))
    bands_number = len(first_circulant_shifted)
    for index in range(bands_number - 2):
        concatenate_hypercube = np.dstack((concatenate_hypercube, first_circulant_shifted[index + 2]))
    return concatenate_hypercube


def optical_water_types_concatenate(symmetrical_derivative_owt):
    symmetrical_derivative_owt_concatenate = np.vstack((symmetrical_derivative_owt[0], symmetrical_derivative_owt[1]))
    length_symmetrical_owt = len(symmetrical_derivative_owt)
    for index in range(length_symmetrical_owt - 2):
        symmetrical_derivative_owt_concatenate = np.vstack((symmetrical_derivative_owt_concatenate,
                                                            symmetrical_derivative_owt[index + 2]))
    return symmetrical_derivative_owt_concatenate


def dataframe_each_row_to_list(hyperspectral_owt_amplitude):
    list_each_row_owt = hyperspectral_owt_amplitude.values.tolist()
    return list_each_row_owt


def dataframe_filter_amplitude(hyperspectral_owt, labels_amplitude):
    hyperspectral_owt_amplitude = hyperspectral_owt[labels_amplitude]
    list_each_row_owt = dataframe_each_row_to_list(hyperspectral_owt_amplitude)
    return list_each_row_owt


def list_data_representation(hyperspectral_clean_data):
    bands_number = hyperspectral_clean_data.shape[-1]
    list_hypercube_data = []
    for index in range(bands_number):
        list_hypercube_data.append(hyperspectral_clean_data[:, :, index])
    return list_hypercube_data


def remaining_rows_list_generator(kernel_size):
    remaining_rows = math.ceil(kernel_size / 2)
    remaining_rows_list = list(range(-remaining_rows, 1))
    remaining_rows_list.sort(reverse=True)
    return remaining_rows_list


def concatenate_bands(first_derivative_list, hyperspectral_clean_data, step_length):
    remaining_list = remaining_rows_list_generator(step_length)
    first_derivative_list_concatenate = first_derivative_list.copy()
    for index in remaining_list:
        first_derivative_list_concatenate.insert(index, hyperspectral_clean_data[:, :, index])
    return first_derivative_list_concatenate


def duplicate_initial_and_final_band(hyperspectral_clean_data, step_length):
    remaining_bands = remaining_rows_list_generator(step_length)
    hyperspectral_list = list_data_representation(hyperspectral_clean_data)
    hypercube_duplicated_list = hyperspectral_list.copy()
    for index in remaining_bands:
        hypercube_duplicated_list.insert(index, hyperspectral_clean_data[:, :, index])
    hypercube_duplicated_bands = hypercube_representation(hypercube_duplicated_list)
    return hypercube_duplicated_bands


def duplicate_initial_and_final_band_owt(hyperspectral_owt, step_length):
    remaining_bands_owt = remaining_rows_list_generator(step_length)
    optical_water_types_duplicated_list = hyperspectral_owt.copy()
    for index in remaining_bands_owt:
        optical_water_types_duplicated_list.insert(index, hyperspectral_owt[index])
    return optical_water_types_duplicated_list


def first_derivative_owt(optical_water_types_list_upper, optical_water_types_list_lower):
    return optical_water_types_list_upper - optical_water_types_list_lower


class SpectralDerivative:
    def __init__(self, hyperspectral_clean_data: np.array, step_length: int):
        self.hyperspectral_clean_data = hyperspectral_clean_data
        self.step_length = step_length

    def spectral_first_order_derivative(self):
        hyperspectral_clean_data_duplicated = duplicate_initial_and_final_band(self.hyperspectral_clean_data,
                                                                               self.step_length)
        bands_number = hyperspectral_clean_data_duplicated.shape[-1]
        first_derivative = []
        for index in range(bands_number - 2):
            numerator_derivative = hyperspectral_clean_data_duplicated[:, :, index] - \
                                   hyperspectral_clean_data_duplicated[:, :, index + 2]
            derivative_operation = numerator_derivative / self.step_length
            first_derivative.append(derivative_operation)
        concatenate_first_derivative = concatenate_bands(first_derivative, self.hyperspectral_clean_data,
                                                         self.step_length)
        hypercube_first_derivative = hypercube_representation(concatenate_first_derivative)
        return hypercube_first_derivative

    def symmetrical_first_order_derivative(self):
        hyperspectral_clean_data_duplicated = duplicate_initial_and_final_band(self.hyperspectral_clean_data,
                                                                               self.step_length)
        bands_number = hyperspectral_clean_data_duplicated.shape[-1]
        symmetrical_derivative = []
        for index in range(1, bands_number - self.step_length):
            absolute_value = np.absolute(index - self.step_length)
            numerator_derivative = hyperspectral_clean_data_duplicated[:, :, index + self.step_length] - \
                                   hyperspectral_clean_data_duplicated[:, :, absolute_value]
            # denominator_derivative = 2 * self.step_length
            denominator_derivative = 20
            derivative_estimation = numerator_derivative / denominator_derivative
            symmetrical_derivative.append(derivative_estimation)
        hypercube_symmetrical_first_derivative = hypercube_representation(symmetrical_derivative)
        return hypercube_symmetrical_first_derivative

    def symmetrical_first_order_derivative_owt(self, hyperspectral_optical_water_type, labels_amplitude):
        hyperspectral_owt_amplitude = dataframe_filter_amplitude(hyperspectral_optical_water_type, labels_amplitude)
        optical_water_types_duplicated_list = duplicate_initial_and_final_band_owt(hyperspectral_owt_amplitude,
                                                                                   self.step_length)
        length_owt = len(optical_water_types_duplicated_list)
        symmetrical_derivative_owt = []
        for index in range(1, length_owt - self.step_length):
            absolute_value = np.absolute(index - self.step_length)
            numerator_derivative = list(map(first_derivative_owt,
                                            optical_water_types_duplicated_list[index + self.step_length],
                                            optical_water_types_duplicated_list[absolute_value]))
            denominator_derivative = 20
            derivative_owt = [index_row / denominator_derivative for index_row in numerator_derivative]
            symmetrical_derivative_owt.append(derivative_owt)
        symmetrical_derivative_owt_concatenate = optical_water_types_concatenate(symmetrical_derivative_owt)
        return symmetrical_derivative_owt_concatenate
