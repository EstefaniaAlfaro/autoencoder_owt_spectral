from MetricsPerformanceSAD import *
from NSAEU_SU_Encoder import *
from ReconstructPatchesDataset import *
from ReshapeFractionalAbundancesMaps import *
from MetricsPerformance import *
from SaveData import *
# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt


DIRECTORY_MODEL = '../model'
SAVE_MODEL_EXTENSION = '.h5'
RESULTS_PATH = '../results'
OPTIMIZER = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.003, decay=0.0)


def experiment_non_su(data, hyperspectral_images, parameters, image_width,
                                                        image_height, abundances_ground_truth,
                                                        ground_truth_end_members):
    root_mean_square_error_append = []
    abundances_maps_reconstructed_append = []
    decoded_data_reconstructed_append = []
    end_members_extracted_append = []
    for index in range(parameters['runs_number']):
        print('Run :----', index)
        print(data.shape)
        print(data.T.shape)
        autoencoder_ns_un_mixing =\
            NonSymmetricalEncoderSu(parameters).non_symmetrical_autoencoder_spectral_un_mixing()
        autoencoder_ns_un_mixing.compile(optimizer=OPTIMIZER, loss=spectral_angle_distance)
        autoencoder_ns_un_mixing.fit(data.T, data.T,
                                        epochs=parameters['epochs'],
                                        shuffle=True,
                                        batch_size=parameters['batch_size_model'])
        abundances_maps = NonSymmetricalEncoderSu(parameters).get_abundances_maps(
            tf.transpose(tf.expand_dims(hyperspectral_images, 0)), autoencoder_ns_un_mixing)
        print(abundances_maps.shape)
        end_members = NonSymmetricalEncoderSu(parameters).get_end_members(tf.transpose(
            tf.expand_dims(hyperspectral_images, 0)), autoencoder_ns_un_mixing)
        print('----------*----------')
    return abundances_maps_reconstructed_append, end_members_extracted_append
