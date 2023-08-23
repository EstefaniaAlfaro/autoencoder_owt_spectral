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
mpl.use('TkAgg')


DIRECTORY_MODEL = '../model'
SAVE_MODEL_EXTENSION = '.h5'
RESULTS_PATH = '../results'


def running_experiment_custom_reduce(data, parameters, enhancement_initializer_mean,
                                     enhancement_initializer_standard_deviation, image_width, image_height,
                                     end_members_number, batch_size_depth, abundances_ground_truth,
                                     ground_truth_end_members):
    root_mean_square_error_append = []
    abundances_maps_reconstructed_append = []
    decoded_data_reconstructed_append = []
    end_members_extracted_append = []
    ordered_abundances_append = []
    for index in range(parameters['runs_number']):
        print('Run :----', index)
        model_autoencoder_custom = experiments_autoencoder_reduce(parameters, enhancement_initializer_mean,
                                                                  enhancement_initializer_standard_deviation)
        model_autoencoder_custom.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.003),
                                         loss=spectral_angle_distance)
        model_autoencoder_custom.fit(data, data,
                                     epochs=parameters['epochs'], shuffle=True,
                                     batch_size=parameters['batch_size_model'])
        decoded_predicted = model_autoencoder_custom.predict(data)
        print('Perform the abundances maps extraction:----')
        feature_extraction = keras.Model(inputs=model_autoencoder_custom.input,
                                         outputs=model_autoencoder_custom.get_layer('tf.nn.softmax').output)
        feature_extraction_layer = feature_extraction(data)
        print('Abundances extraction is done:----')
        get_end_members_weights = model_autoencoder_custom.get_layer('end_members').get_weights()
        # get_end_members_weights = model_autoencoder_custom.get_layer('end_members').get_weights()[0]
        end_members_extracted = get_end_members(get_end_members_weights)
        print('Training the model is done: ----')
        abundances_maps_reconstructed = ReconstructTrainingPatches(feature_extraction_layer.numpy(),
                                                                   image_width,
                                                                   image_height,
                                                                   end_members_number).reconstruct_data_patches()
        decoded_data_reconstructed = ReconstructTrainingPatches(decoded_predicted,
                                                                image_width, image_height,
                                                                batch_size_depth).reconstruct_data_patches()
        abundances_maps_reconstructed_append.append(abundances_maps_reconstructed)
        decoded_data_reconstructed_append.append(decoded_data_reconstructed)
        end_members_extracted_append.append(end_members_extracted)
        print('4. Saving abundances maps, decoded_data, and end-members, hyper-parameters: ----')
        SaveDataPreprocess(RESULTS_PATH, parameters["data_preparation_title"] + '_run_' + str(index),
                           data=[abundances_maps_reconstructed, decoded_data_reconstructed, parameters,
                                 end_members_extracted],
                           label=['abundances_maps', 'decoder', 'parameters', 'end_members_extracted']) \
            .save_dictionary_data()
        root_mean_square_error, ordered_abundances = \
            MetricsPerformance(abundances_ground_truth, abundances_maps_reconstructed).metric_root_mean_square_error()
        root_mean_square_error_append.append(root_mean_square_error)
        ordered_abundances_append.append(ordered_abundances)
        spectral_angle_metrics, ordered_end_members = MetricsPerformanceSAD(ground_truth_end_members,
                                                                            end_members_extracted).metrics_sad()
        print('4. Save in a Dictionary the best fit between the reconstructed abundance, and the ground truth: ----')
        abundances_fit_ground_reconstruction_dictionary = BestFitReconstructionAbundances(abundances_ground_truth,
                                                                                          abundances_maps_reconstructed,
                                                                                          ordered_abundances).\
            extract_best_fit_abundances()
        print('5. Save in a Dictionary the best fit between the reconstructed end_member, and the ground truth: ----')
        end_members_fit_ground_reconstruction_dictionary = BestFitReconstructionEndMembers(ground_truth_end_members,
                                                                                           end_members_extracted,
                                                                                           ordered_end_members).\
            extract_best_fit_end_member()
        SaveDictionaryFormat(abundances_fit_ground_reconstruction_dictionary, parameters["title_dictionary_ab"] + '_' +
                             str(index)).save_dictionary_format()
        SaveDictionaryFormat(end_members_fit_ground_reconstruction_dictionary, parameters["title_dictionary_end"] +
                             '_' + str(index)).save_dictionary_format()
        print('6. Printing metrics abundances maps: ----')
        ShowMetricsResults(ordered_abundances, parameters["data_preparation_title"] + '_' + 'rmse_abundances_' +
                                                          str(index), parameters["abundances_title"])\
            .save_metrics_results_abundances()
        ShowMetricsResults(ordered_end_members, parameters["data_preparation_title"] + '_' + 'Sad_end_members_' +
                           str(index), parameters["end_member_title"]).save_metrics_results_abundances()
        print('6. Saving model: ----')
        model_autoencoder_custom.save(DIRECTORY_MODEL + '/autoencoder_model_spectral_' + str(index) +
                                      SAVE_MODEL_EXTENSION)
        print('7. Program finish :)')
        print('Additional step once the test is performed this lines will be removed')
        data_dec = tf.squeeze(decoded_data_reconstructed)
        data_tra = tf.squeeze(data)
        plt.imshow(data_dec[:, :, 10].numpy())
        plt.show()
        plt.figure()
        plt.imshow(data_tra[:, :, 10].numpy())
        plt.show()
        print('Additional step once the test is performed this lines will be removed')
        del model_autoencoder_custom, decoded_predicted, feature_extraction, get_end_members_weights, \
            end_members_extracted
        print('----------*----------')
    return end_members_extracted_append, decoded_data_reconstructed_append, abundances_maps_reconstructed_append
