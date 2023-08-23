from SaveData import *
from NSAEU_Encoder import *
from NSAEU_Model import *

DIRECTORY_MODEL = '../model'
SAVE_MODEL_EXTENSION = '.h5'
RESULTS_PATH = '../results'
OPTIMIZER = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=0.0)


def experiment_abundances_then_end_members(data, hyperspectral_images, parameters, image_width,
                                                        image_height, abundances_ground_truth,
                                                        ground_truth_end_members):
    abundances_maps_reconstructed_append = []
    decoded_data_reconstructed_append = []
    end_members_extracted_append = []
    for index in range(parameters['runs_number']):
        print('Run :----', index)
        print('original shape:', data.shape)
        print('data transpose:', tf.transpose(data).shape)
        convolutional_non_symmetrical = ModelNSAutoencoder(parameters)
        convolutional_non_symmetrical.compile(optimizer=OPTIMIZER, loss=tf.keras.losses.CategoricalCrossentropy())
        convolutional_non_symmetrical.fit(data, data,
                                          epochs=parameters['epochs'],
                                          shuffle=True,
                                          batch_size=parameters['batch_size_model'])
        convolutional_non_symmetrical.nsau_encoder.summary()
        end_members_extracted = convolutional_non_symmetrical.nsau_encoder(data).numpy()
        abundances_maps_estimation = convolutional_non_symmetrical.abundances_extracted()
        decoded_data = convolutional_non_symmetrical.nsau_decoder(end_members_extracted)
        print(abundances_maps_estimation.shape)
        print(end_members_extracted.shape)
        print('Training the model is done :): ----')
        SaveDataPreprocess(RESULTS_PATH, parameters["data_preparation_title"] + '_run_' + str(index),
                           data=[abundances_maps_estimation, end_members_extracted, decoded_data],
                           label=['abundances_maps', 'end_member', 'decoded_data']).save_dictionary_data()
    return abundances_maps_reconstructed_append, end_members_extracted_append
