from MetricsPerformanceSAD import *
from DCNN_SU_Model import *
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
OPTIMIZER = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001, decay=0.0)


def running_experiment_deep_non_symmetrical_autoencoder(data, hyperspectral_images, parameters, image_width,
                                                        image_height, abundances_ground_truth,
                                                        ground_truth_end_members):
    abundances_maps_reconstructed_append = []
    decoded_data_reconstructed_append = []
    end_members_extracted_append = []
    for index in range(parameters['runs_number']):
        print('Run :----', index)
        print(data.shape)
        print(data.T.shape)
        autoencoder_non_symmetrical = DeepConvolutionalModel(parameters)
        autoencoder_non_symmetrical.compile(optimizer=OPTIMIZER, loss=spectral_angle_distance)
        autoencoder_non_symmetrical.fit(data.T, data.T,
                                        epochs=parameters['epochs'],
                                        shuffle=True,
                                        batch_size=parameters['batch_size_model'])
        autoencoder_non_symmetrical.encoder_ns.summary()
        end_members_extracted = autoencoder_non_symmetrical.encoder_ns(tf.transpose(tf.expand_dims(hyperspectral_images,
                                                                                                   0))).numpy()
        end_members_extraction_prediction = autoencoder_non_symmetrical.end_members_extraction(hyperspectral_images)
        decoded_data = autoencoder_non_symmetrical.decoder_ns(end_members_extracted).numpy()
        abundances_maps_estimation = autoencoder_non_symmetrical.abundances_maps_extraction()
        print('Training the model is done: ----')
        abundances_maps_reshape = ReshapeFractionalMaps(abundances_maps_estimation, image_width, image_height,
                                                        parameters).reshape_abundances_maps()
        abundances_maps_reconstructed = ReconstructTrainingPatches(abundances_maps_reshape,
                                                                   image_width,
                                                                   image_height,
                                                                   parameters["number_end_members"])\
            .reconstruct_data_patches()
        decoded_data_reconstructed = ReconstructTrainingPatches(decoded_data.T,
                                                                image_width,
                                                                image_height,
                                                                parameters["number_bands"]).reconstruct_data_patches()
        abundances_maps_reconstructed_append.append(abundances_maps_reconstructed)
        end_members_extracted_append.append(end_members_extracted)
        decoded_data_reconstructed_append.append(decoded_data_reconstructed)
        print('4. Saving abundances maps, decoded_data, and end-members, hyper-parameters: ----')
        SaveDataPreprocess(RESULTS_PATH, parameters["data_preparation_title"] + '_run_' + str(index),
                           data=[abundances_maps_reconstructed, decoded_data_reconstructed, parameters,
                                 end_members_extracted],
                           label=['abundances_maps', 'decoder', 'parameters', 'end_members_extracted']) \
            .save_dictionary_data()
        print('6. Saving model: ----')
        autoencoder_non_symmetrical.save_weights(DIRECTORY_MODEL + '/autoencoder_model_spectral_' + str(index) +
                                                 SAVE_MODEL_EXTENSION)
        print('Additional step once the test is performed this lines will be removed')
        del autoencoder_non_symmetrical, decoded_data, abundances_maps_estimation, \
            end_members_extracted
        print('----------*----------')
    return abundances_maps_reconstructed_append, end_members_extracted_append
