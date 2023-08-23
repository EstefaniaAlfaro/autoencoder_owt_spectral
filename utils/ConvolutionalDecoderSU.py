import tensorflow as tf


class DecoderSU(tf.keras.layers.Layer):
    def __init__(self, parameters):
        super(DecoderSU, self).__init__()
        self.decoder_layer = tf.keras.layers.Conv2D(filters=parameters['filter_3'],
                                                    kernel_size=parameters['size_3'],
                                                    activation='linear',
                                                    kernel_constraint=tf.keras.constraints.non_neg(),
                                                    name='end_members_weights',
                                                    strides=1,
                                                    padding='same',
                                                    kernel_regularizer=None,
                                                    kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.3),
                                                    use_bias=False)

    def call(self, transformation):
        reconstruction = self.decoder_layer(transformation)
        return reconstruction

    def end_members_extraction(self):
        end_members = self.decoder_layer.get_weights()
        return end_members

