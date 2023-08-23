import pdb
import numpy as np
import pandas as pd
import math


def hypercube_to_dataframe(hyperspectral_data_clean):
    bands_number = hyperspectral_data_clean.shape[-1]
    width_size = hyperspectral_data_clean.shape[0]
    height_size = hyperspectral_data_clean.shape[1]
    hypercube_data_reshape = np.zeros((width_size * height_size, bands_number))
    for index in range(bands_number):
        hypercube_data_reshape[:, index] = np.reshape(hyperspectral_data_clean[:, :, index], -1)
    hypercube_reshape_dataframe_transpose = dataframe_transformation(hypercube_data_reshape)
    return hypercube_reshape_dataframe_transpose


def average_kernel_operation(hypercube_chunks_by_row):
    return hypercube_chunks_by_row.mean()


def remaining_rows_list_generator(kernel_size):
    remaining_rows = math.ceil(kernel_size / 2)
    remaining_rows_list = list(range(-remaining_rows, 1))
    remaining_rows_list.sort(reverse=True)
    return remaining_rows_list


def concatenate_remaining_rows(hypercube_reshape_dataframe_selected_rows, hypercube_chunk_average, kernel_size):
    remaining_row_data = remaining_rows_list_generator(kernel_size)
    hypercube_chunk_average_copy = hypercube_chunk_average.copy()
    middle_point_kernel = math.floor(kernel_size / 2)
    for index_row in remaining_row_data:
        hypercube_chunk_average_copy.insert(index_row, hypercube_reshape_dataframe_selected_rows.iloc[
            middle_point_kernel, index_row])
    return hypercube_chunk_average_copy


def adding_initial_and_final_bands(hypercube_reshape_dataframe_transpose, hypercube_concatenate_data, kernel_size):
    remaining_rows_list = remaining_rows_list_generator(kernel_size)
    hypercube_concatenate_data_copy = hypercube_concatenate_data.copy()
    for index in remaining_rows_list:
        hypercube_concatenate_data_copy.insert(index, hypercube_reshape_dataframe_transpose.iloc[index, :].values)
    return hypercube_concatenate_data_copy


def concatenate_stack_filter_hypercube(hypercube_concatenate_data_copy, bands_number):
    hypercube_concatenate_stack = np.vstack((hypercube_concatenate_data_copy[0], hypercube_concatenate_data_copy[1]))
    for index in range(bands_number - 2):
        hypercube_concatenate_stack = np.vstack((hypercube_concatenate_stack, hypercube_concatenate_data_copy[index+2]))
    return hypercube_concatenate_stack


def hypercube_reshape_original_size(hypercube_complete_depth_stack, hyperspectral_data_clean):
    rows_length = hypercube_complete_depth_stack.shape[0]
    width_image = hyperspectral_data_clean.shape[0]
    height_image = hyperspectral_data_clean.shape[1]
    hypercube_reshape_size = []
    for index in range(rows_length):
        hypercube_reshape_size.append(np.reshape(hypercube_complete_depth_stack[index], (width_image, height_image)))
    return hypercube_reshape_size


def depth_concatenate_hypercube(hypercube_reshape_size):
    row_length = len(hypercube_reshape_size)
    depth_concatenate_filtered_hypercube = np.dstack((hypercube_reshape_size[0], hypercube_reshape_size[1]))
    for index in range(row_length - 2):
        depth_concatenate_filtered_hypercube = np.dstack((depth_concatenate_filtered_hypercube,
                                                          hypercube_reshape_size[index + 2]))
    return depth_concatenate_filtered_hypercube


def average_filter_row(hypercube_reshape_dataframe_transpose, kernel_size):
    bands_number = hypercube_reshape_dataframe_transpose.shape[0]
    row_length = hypercube_reshape_dataframe_transpose.shape[-1]
    concatenate_average_hypercube = []
    for row_index in range(bands_number - kernel_size):
        hypercube_chunks_by_row = hypercube_reshape_dataframe_transpose.iloc[row_index:kernel_size + row_index, :]
        hypercube_chunk_average = list(
            map(average_kernel_operation, [hypercube_chunks_by_row.iloc[:, index:kernel_size + index].values
                                           for index in range(row_length - kernel_size)]))
        concatenate_average_data = concatenate_remaining_rows(hypercube_chunks_by_row, hypercube_chunk_average,
                                                              kernel_size)
        concatenate_average_hypercube.append(concatenate_average_data)
    hypercube_concatenate_complete = adding_initial_and_final_bands(hypercube_reshape_dataframe_transpose,
                                                                    concatenate_average_hypercube, kernel_size)
    hypercube_complete_depth_stack = concatenate_stack_filter_hypercube(hypercube_concatenate_complete, bands_number)
    return hypercube_complete_depth_stack


def dataframe_transformation(hypercube_data_reshape):
    hypercube_data_transformation = pd.DataFrame(data=hypercube_data_reshape)
    hypercube_dataframe_transpose = hypercube_data_transformation.T
    return hypercube_dataframe_transpose


def data_type(hyperspectral_data_clean):
    hypercube_transformation = hyperspectral_data_clean
    if type(hypercube_transformation).__name__ == 'ndarray':
        hypercube_transformation = hypercube_to_dataframe(hyperspectral_data_clean)
    return hypercube_transformation


class MeanSmoothing:
    def __init__(self, hyperspectral_data_clean, kernel_size_mean_filter, padding_mean_filter):
        self.hyperspectral_data_clean = hyperspectral_data_clean
        self.kernel_size_mean_filter = kernel_size_mean_filter
        self.padding_mean_filter = padding_mean_filter

    def smoothing_average_filter(self):
        hypercube_transformation = data_type(self.hyperspectral_data_clean)
        hypercube_complete_filtered_stack = average_filter_row(hypercube_transformation, self.kernel_size_mean_filter)
        filtered_hypercube_list = hypercube_reshape_original_size(hypercube_complete_filtered_stack,
                                                                  self.hyperspectral_data_clean)
        filtered_hypercube = depth_concatenate_hypercube(filtered_hypercube_list)
        return filtered_hypercube
