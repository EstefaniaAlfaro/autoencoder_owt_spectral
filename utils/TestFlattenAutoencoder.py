from PreprocessingStageSpectralUnMixing import *
from SaveConfigurationFile import *
from TestNonSymmetricalAutoencoder import *


data_path = '../data/samson_1.mat'
ground_truth_path = '../data/end3.mat'
results_path = '../results/'
data_target = 'V'
ground_truth_target = 'M'
ground_truth_abundances_label = 'A'
extension = '.mat'
patch_size = 32  # Size of the patches to be extracted from the input images
image_size = 95
image_width = 95
image_height = 95
batch_size = 20
rows_number = 10
columns_number = 10
batch_size_depth = 156
training_percentage = 70
testing_percentage = 20
validation_percentage = 10
number_images = 8
end_members_number = 4
case_option_abundances_maps = 0
number_patches_width = 3
number_patches_height = 3
patch_size_model = 40
number_patches = 250
batch_size_model = 15
# configuration_file = 'configuration_file_DCNN_SU'
configuration_file = 'samson_configuration_file_DCNN_SU'


def blind_autoencoder_test():
    print('1. Loading the configuration file: ----')
    # SaveConfigurationFile(results_path, configuration_file).save_configurations_file(parameters_file)
    parameters = SaveConfigurationFile(results_path, configuration_file).load_configurations_file()
    print('2. Starting data preparation: ----')
    stack_patches_data, enhancement_initializer_energy, enhancement_initializer_mean, \
    enhancement_initializer_standard_deviation, reshape_abundances_maps_tensor, \
    training_data, testing_data, validation_data, random_position, hypercube_data_patches, abundances_ground_truth,\
        ground_truth_end_members, hypercube_data = preprocessing_stage_spectral(data_path, ground_truth_path,
                                                                                data_target, ground_truth_target,
                                                                                ground_truth_abundances_label,
                                                                                image_width, image_height,
                                                                                end_members_number,
                                                                                case_option_abundances_maps,
                                                                                batch_size, rows_number, columns_number,
                                                                                batch_size_depth, patch_size,
                                                                                image_size,
                                                                                training_percentage, testing_percentage,
                                                                                validation_percentage,
                                                                                parameters['patch_size_model'],
                                                                                number_patches, batch_size_model)
    print('Data preparation is done: ----')
    print('3. Training the model: ----')
    abundances_maps_reconstructed_append, end_members_extracted_append = \
        experiment_abundances_then_end_members(hypercube_data_patches, hypercube_data, parameters,
                                                            image_width, image_height, abundances_ground_truth,
                                                            ground_truth_end_members)
    print('---- Program finish :)----')


blind_autoencoder_test()
