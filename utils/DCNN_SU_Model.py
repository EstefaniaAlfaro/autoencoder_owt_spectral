from DCNN_SU_Decoder import *
from DCNN_SU_Encoder import *
from SumToOneConstraint import *


class DeepConvolutionalModel(tf.keras.Model):
    def __init__(self, parameters):
        self.parameters = parameters
        super(DeepConvolutionalModel, self).__init__()
        self.encoder_ns = DeepConvolutionalEncoder(parameters)
        self.decoder_ns = DeepConvolutionalDecoder(parameters)

    def call(self, data):
        get_end_members = self.encoder_ns(data)
        decoder = self.decoder_ns(get_end_members)
        return decoder

    def abundances_maps_extraction(self):
        abundances_maps = self.decoder_ns.get_weights()[0]
        return abundances_maps

    def end_members_extraction(self, hyperspectral_images):
        end_members = self.encoder_ns.predict(tf.transpose(tf.expand_dims(hyperspectral_images, 0)))
        return end_members
