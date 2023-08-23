from ReconstructPatchesDataset import *
from MetricsPerformanceSAD import *
import matplotlib as mpl
from BlindAutoencoderSU import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')


DIRECTORY_MODEL = '../model'
SAVE_MODEL_EXTENSION = '.h5'
RESULTS_PATH = '../results'
OPTIMIZER = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.003)


def running_experiment_blind_autoencoder_su(data, hyperspectral_images, parameters, image_width, image_height):
    root_mean_square_error_append = []
    abundances_maps_reconstructed_append = []
    decoded_data_reconstructed_append = []
    end_members_extracted_append = []
    ordered_abundances_append = []
    for index in range(parameters['runs_number']):
        print('Run :----', index)
        blind_autoencoder_su = BlindAutoencoderSU(parameters)
        blind_autoencoder_su.compile(optimizer=OPTIMIZER, loss=spectral_angle_distance)
        blind_autoencoder_su.fit(data.T, data.T,
                                 epochs=parameters['epochs'],
                                 shuffle=True,
                                 batch_size=parameters['batch_size_model'])
        blind_autoencoder_su.blind_encoder_su .summary()
        blind_autoencoder_su.blind_decoder_su.summary()
        encoded_data = blind_autoencoder_su.blind_encoder_su(tf.transpose(tf.expand_dims(hyperspectral_images,
                                                                                         0))).numpy()
        decoded_data = blind_autoencoder_su.blind_decoder_su(encoded_data).numpy()
        abundances_maps = blind_autoencoder_su.get_abundances_maps()
    return abundances_maps_reconstructed_append, end_members_extracted_append
