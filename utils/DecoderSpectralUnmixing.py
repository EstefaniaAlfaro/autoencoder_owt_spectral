import keras
from tensorflow.keras import layers, losses
import tensorflow as tf


class DecoderSpectralUnMixing(tf.keras.Model):
    def __init__(self):
        super(DecoderSpectralUnMixing, self).__init__()
        self.layer_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, padding='same', strides=1, activation='relu')
        self.layer_4 = tf.keras.layers.Conv2DTranspose(64, kernel_size=1, padding='same')
        self.layer_5 = tf.keras.layers.Conv2DTranspose(3, kernel_size=1, padding='same',
                                                       activation=tf.keras.activations.softmax, name='abundances_maps')
        self.decoder = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid', padding='same')

    def call(self, encoder):
        decoder = self.layer_3(encoder)
        decoder = self.layer_4(decoder)
        decoder = self.layer_5(decoder)
        decoder = self.decoder(decoder)
        return decoder

    def end_members_extraction(self):
        return self.decoder.get_weights()
