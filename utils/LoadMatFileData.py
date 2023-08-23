import scipy.io
import os


class LoadDataMatlabFile:
    def __init__(self, path, data):
        self.path = path
        self.data = data

    def load_matlab_file_extension(self):
        full_path_dictionary = os.path.join(self.path, self.data)
        dataset = scipy.io.loadmat(full_path_dictionary)
        return dataset
