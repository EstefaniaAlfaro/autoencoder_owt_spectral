from ReadHdrImagesFiles import *
import os

PATH_DATA = '../data'


def load_swath_tracks(parameters):
    full_path_image = os.path.join(PATH_DATA, parameters["hsi2.0_image"])
    full_path_image_hdr = os.path.join(PATH_DATA, parameters["hsi2.0_hdr"])
    load_swath_tracks_high_dynamic_range = ReadHdrImageFiles(full_path_image, full_path_image_hdr)\
        .read_high_dynamic_range_files()
    return load_swath_tracks_high_dynamic_range
