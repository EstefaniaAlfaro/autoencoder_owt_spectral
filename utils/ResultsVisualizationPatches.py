from LoadMatFileData import *
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import numpy as np
from scipy.io import savemat
from ReadGeoTifFiles import *


PATH_RESULTS = '../data'
filename = 'data_conf_jasper_run_10_run_0.mat'
number_bands = 198
# number_bands = 156
number_end_members = 4


def results_visualization_patches():
    # hypercube_data = ReadGeoTifFiles('hab_yellow_roi.tif').read_geo_tif_file()
    # image_width = hypercube_data.shape[1]
    # image_height = hypercube_data.shape[2]
    image_width = 100
    image_height = 100
    data = LoadDataMatlabFile(PATH_RESULTS, filename).load_matlab_file_extension()
    abundances = data['abundances_maps']
    end_members = data['end_member']
    decoder = data['decoded_data']
    reconstructed_data_abundances = reconstruct_from_patches_2d(np.array(end_members), (image_width, image_height,
                                                                                        number_end_members))
    reconstructed_end_members = np.squeeze(abundances).mean(axis=0).mean(axis=0)
    data_jasper_run_10 = {"endmembers_jasper_10": reconstructed_end_members,
                            "abundances_jasper_10": reconstructed_data_abundances}
    savemat(os.path.join("..", "data", 'jasper_run_10.mat'), data_jasper_run_10)
    # data_hab = {'hab_end_members_green_roi': reconstructed_end_members,
    #                     'abundances_green_roi': reconstructed_data_abundances}
    # savemat('data_green_roi.mat', data_hab)


results_visualization_patches()
