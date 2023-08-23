from MetricsPerformanceSAD import *
from BlindAutoencoderSU import *
from DCNN_SU import *


DIRECTORY_MODEL = '../model'
SAVE_MODEL_EXTENSION = '.h5'
RESULTS_PATH = '../results'
OPTIMIZER = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.003)


def running_experiment_deep_autoencoder_su(data, hyperspectral_images, parameters, image_width, image_height):
    root_mean_square_error_append = []
    abundances_maps_reconstructed_append = []
    decoded_data_reconstructed_append = []
    end_members_extracted_append = []
    ordered_abundances_append = []
    hyperspectral_images = hyperspectral_images.astype('float32') / 255
    for index in range(parameters['runs_number']):
        print('Run :----', index)
        data = data.astype('float32') / 255
        autoencoder_deep_cnn_su = DeepConvolutionalSpectralUnMixing(parameters)
        autoencoder_deep_cnn_su.compile(optimizer=OPTIMIZER, loss=spectral_angle_distance)
        autoencoder_deep_cnn_su.fit(data.T, data.T,
                                    epochs=parameters['epochs'],
                                    shuffle=True,
                                    batch_size=parameters['batch_size_model'])
        autoencoder_deep_cnn_su.deep_encoder_su.summary()
        autoencoder_deep_cnn_su.deep_decoder_su.summary()
        encoded_data = autoencoder_deep_cnn_su.deep_encoder_su(tf.transpose(tf.expand_dims(hyperspectral_images,
                                                                                           0))).numpy()
        decoded_data = autoencoder_deep_cnn_su.deep_decoder_su(encoded_data).numpy()
        end_members_extracted = autoencoder_deep_cnn_su.end_members_extraction(hyperspectral_images)
        abundances_maps = autoencoder_deep_cnn_su.abundances_maps_extraction()
    return abundances_maps_reconstructed_append, end_members_extracted_append
