from spectral import *
import os

PATH_RESULTS = '../data'


class ReadHdrImageData:
    def __init__(self, parameters_hdr, parameters_image):
        self.parameters_hdr = parameters_hdr
        self.parameters_image = parameters_image

    def read_high_dynamic_range_files(self):
        full_path_data_hdr = os.path.join(PATH_RESULTS, self.parameters_hdr)
        full_path_data = os.path.join(PATH_RESULTS, self.parameters_image)
        high_dynamic_range = envi.open(full_path_data_hdr, full_path_data)
        load_spatial_content_high_dynamic_range = high_dynamic_range.open_memmap()
        return load_spatial_content_high_dynamic_range
