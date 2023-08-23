from ExperimentsAutoencoderReduce import *
from ReconstructPatchesDataset import *
from MetricsPerformance import *
from SaveData import *
from ShowResults import *
from MetricsPerformanceSAD import *
from BestReconstructionAbundances import *
from BestReconstructionEndMembers import *
from SaveDictionaryFormat import *
from matplotlib import pyplot as plt
import matplotlib as mpl
from BlindAutoencoder import *
import numpy as np

mpl.use('TkAgg')

DIRECTORY_MODEL = '../model'
SAVE_MODEL_EXTENSION = '.h5'
RESULTS_PATH = '../results'
OPTIMIZER = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.003)


def running_experiment_blind_autoencoder(data, hyperspectral_images, parameters, image_width, image_height):
    root_mean_square_error_append = []
    abundances_maps_reconstructed_append = []
    decoded_data_reconstructed_append = []
    end_members_extracted_append = []
    ordered_abundances_append = []
    for index in range(parameters['runs_number']):
        print('Run :----', index)
        blind_autoencoder = BlindAutoencoder(parameters)
        blind_autoencoder.compile(optimizer=OPTIMIZER, loss=total_loss_function)
        blind_autoencoder.fit(data, data,
                              epochs=parameters['epochs'],
                              shuffle=True,
                              batch_size=parameters['batch_size_model'])
        blind_autoencoder.blind_encoder.summary()
        blind_autoencoder.blind_decoder.summary()
        abundances_maps = blind_autoencoder.get_abundances(hyperspectral_images)
        end_members_extracted = blind_autoencoder.get_end_members()
        encoder_images = blind_autoencoder.blind_encoder(data)
        decoded_reconstruction = blind_autoencoder.blind_decoder(encoder_images)
        abundances_maps_reconstructed = ReconstructTrainingPatches(abundances_maps,
                                                                   image_width,
                                                                   image_height,
                                                                   parameters['number_end_members']).\
            reconstruct_data_patches()
    return abundances_maps_reconstructed_append, end_members_extracted_append
