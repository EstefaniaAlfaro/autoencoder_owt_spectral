import scipy.io
import os

PATH_DATA = '../data'


def load_dataset(dictionary_read_matlab_image):
    list_keys = list(dictionary_read_matlab_image.keys())[-1]
    dataset = dictionary_read_matlab_image[list_keys]
    return dataset


class MatLabFile:
    def __init__(self, parameters):
        self.parameters = parameters

    def read_matlab_file(self):
        full_path = os.path.join(PATH_DATA, self.parameters)
        read_matlab_image = scipy.io.loadmat(full_path)
        load_data_matlab_file = load_dataset(read_matlab_image)
        return load_data_matlab_file

