import numpy as np
from tensorflow.keras import Model, Sequential, layers, optimizers, activations
import tensorflow as tf
from ConvolutionalEncoderSU import *
from ConvolutionalDecoderSU import *


class AutoencoderSu(tf.keras.Model):
    def __init__(self, parameters):
        super(AutoencoderSu, self).__init__()
        self.encoder_su = EncoderSU(parameters)
        self.decoder_su = DecoderSU(parameters)
        self.parameters = parameters

    def call(self, data):
        abundances = self.encoder_su(data)
        reconstruction = self.decoder_su(abundances)
        return reconstruction

    def end_members_extraction(self):
        end_members = self.decoder_su.end_members_extraction()[0]
        if end_members.shape[-1] > 1:
            end_members = np.squeeze(end_members).mean(axis=0).mean(axis=0)
        else:
            end_members = np.squeeze(end_members)
        return end_members

    def abundances_extraction(self, data):
        abundances = np.squeeze(self.encoder_su.predict(np.expand_dims(data, 0)))
        return abundances

    def train(self, data):
        self.fit(data, data,
                 epochs=self.parameters['epochs'],
                 batch_size=self.parameters['batch_size'],
                 verbose=0)

