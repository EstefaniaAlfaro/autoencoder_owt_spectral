import keras.metrics

from AutoencoderSpectralIUnmixing import *
from DecoderVisualization import *
from GradientEstimation import *
from CustomLossFunctions import *


def experiment_custom_autoencoder(stack_patches_data, enhancement_initializer_mean,
                                  enhancement_initializer_standard_deviation, optimizer, number_images, number_runs):
    stack_patches_data = stack_patches_data[..., tf.newaxis]
    decoded_images_spectral = []
    for index in range(number_runs):
        autoencoder_spectral = AutoencoderSpectralUnMixing(enhancement_initializer_mean,
                                                           enhancement_initializer_standard_deviation)
        autoencoder_spectral.compile(optimizer, loss=focal_loss)
        autoencoder_spectral.fit(stack_patches_data,
                                 stack_patches_data,
                                 epochs=10,
                                 shuffle=True)
        decoded_images_spectral = autoencoder_spectral.predict(stack_patches_data)
        DecoderVisualization(stack_patches_data, decoded_images_spectral).decoded_images(number_images)
        end_members = autoencoder_spectral.end_members_extraction()
    return decoded_images_spectral
