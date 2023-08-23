from LoadMatFileData import *
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import numpy as np
from ReadGeoTifFiles import *
# matplotlib.use('TkAgg')
# from scipy.io import savemat
# from matplotlib import pyplot as plt


# 32 best results data_configuration_conv_32_run_0.mat also 33, but 34 it is perfect, for jasper
# 35 it is awesome it is the best reconstruction 100 epochs for jasper
# 36 it is good for the other endmembers it is the best reconstruction 300 epochs for jasper
# Until now 37 it is the image show at the results for jasper 250 epochs

PATH_RESULTS = '../data'
filename = 'hab_data_roi_yellow_1_run_0.mat'
# number_bands = 198
# number_bands = 156
# number_bands = 31
# image_width = 255
# image_height = 232
number_end_members = 4


def results_visualization_patches():
    hypercube_data = ReadGeoTifFiles('hab_yellow_roi.tif').read_geo_tif_file()
    image_width = hypercube_data.shape[1]
    image_height = hypercube_data.shape[2]
    data = LoadDataMatlabFile(PATH_RESULTS, filename).load_matlab_file_extension()
    abundances = data['abundances_maps']
    end_members = data['end_member']
    decoder = data['decoded_data']
    reconstructed_data_abundances = reconstruct_from_patches_2d(np.array(end_members), (image_width, image_height,
                                                                                        number_end_members))
    reconstructed_end_members = np.squeeze(abundances).mean(axis=0).mean(axis=0)
    # data_hab = {'hab_end_members_green_roi': reconstructed_end_members,
    #                     'abundances_green_roi': reconstructed_data_abundances}
    # savemat('data_green_roi.mat', data_hab)
    print(data)


results_visualization_patches()