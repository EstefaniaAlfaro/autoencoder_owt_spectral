from spectral import *


class ReadHdrImageFiles:
    def __init__(self, full_path_data, full_path_data_hdr):
        self.full_path_data = full_path_data
        self.full_path_data_hdr = full_path_data_hdr

    def read_high_dynamic_range_files(self):
        high_dynamic_range = envi.open(self.full_path_data_hdr, self.full_path_data)
        load_spatial_content_high_dynamic_range = high_dynamic_range.open_memmap()
        return load_spatial_content_high_dynamic_range

