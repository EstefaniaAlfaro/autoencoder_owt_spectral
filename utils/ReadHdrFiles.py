import pdb
import os
import numpy as np
from spectral import *


def complete_path(folder_path, image_path):
    full_path = os.path.join(folder_path, image_path)
    return full_path


class ReadHdrFiles:
    def __init__(self, folder_path, image_path, image_path_header):
        self.image_path = image_path
        self.folder_path = folder_path
        self.image_path_header = image_path_header

    def read_high_dynamic_range_file(self):
        path_data = complete_path(self.folder_path, self.image_path)
        path_data_header = complete_path(self.folder_path, self.image_path_header)
        high_dynamic_range = envi.open(path_data, path_data_header)
        load_spatial_content_high_dynamic = high_dynamic_range.open_memmap()
        return load_spatial_content_high_dynamic
