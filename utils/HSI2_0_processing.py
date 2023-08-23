from ReadGeoTifFiles import *
from SaveConfigurationFile import *
from preprocessing_images_hsi2 import *
from FilterOperationsHSI2 import *
from TestNonSymmetricalAutoencoder import *

results_path = '../results/'
configuration_file = 'configuration_file_hsi2'
number_patches = 250
batch_size_model = 15
abundances_ground_truth = None
ground_truth_end_members = None


def processing_image_hsi2():
    print('1. Loading the configuration file: ----')
    parameters = SaveConfigurationFile(results_path, configuration_file).load_configurations_file()
    print('2. Load the hsi2.0 image: ----')
    hypercube_data = ReadGeoTifFiles(parameters["hsi2_tiff_file"]).read_geo_tif_file()
    print('3. Starting mean smoothing filter and spectral derivative: ----')
    hypercube_spectral_derivative = filter_operations_hsi2(hypercube_data, parameters)
    hypercube_data_patches, end_member_extraction_n_findr_hsi2, abundances_estimation_hsi2, \
    end_member_extraction_fippi_hsi2, abundances_estimation_fippi_hsi2, enhancement_hsi2_initializer_energy, \
    enhancement_hsi2_initializer_mean, enhancement_hsi2_initializer_standard_deviation = \
        preprocessing_image_hsi2(parameters, hypercube_spectral_derivative, number_patches, batch_size_model)
    print('Data preparation is done: ----')
    image_width = hypercube_data.shape[1]
    image_height = hypercube_data.shape[2]
    print('3. Training the model: ----')
    abundances_maps_reconstructed_append, end_members_extracted_append = \
        experiment_abundances_then_end_members(hypercube_data_patches, hypercube_data, parameters,
                                                            image_width, image_height, abundances_ground_truth,
                                                            ground_truth_end_members)
    print('---- Program finish :)----')


processing_image_hsi2()

