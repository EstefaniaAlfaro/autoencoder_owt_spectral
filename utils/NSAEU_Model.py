from tensorflow.keras import layers, losses, regularizers
from SumToOneConstraint import *
from GreaterThanZero import *
import numpy as np
from NSAEU_Encoder import *
from NSAEU_Decoder import *


class ModelNSAutoencoder(tf.keras.Model):
    def __init__(self, parameters):
        super(ModelNSAutoencoder, self).__init__()
        self.parameters = parameters
        self.nsau_encoder = NonSymmetricalConvolutionalEncoder(self.parameters)
        self.nsau_decoder = NonSymmetricalDecoder(self.parameters)

    def call(self, data):
        encoder = self.nsau_encoder(data)
        decoder = self.nsau_decoder(encoder)
        return decoder

    def end_members_extraction(self, hyperspectral_images):
        end_members = tf.squeeze(self.nsau_encoder.predict(tf.expand_dims(hyperspectral_images, 0)))
        return end_members

    def abundances_extracted(self):
        abundances_estimated = self.nsau_decoder.abundances_estimation()
        return abundances_estimated
