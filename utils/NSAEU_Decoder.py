from CustomNormalization import *


class NonSymmetricalDecoder(tf.keras.layers.Layer):
    def __init__(self, parameters):
        super(NonSymmetricalDecoder, self).__init__()
        self.parameters = parameters
        self.custom_normalization = CustomNormalization(self.parameters, name='abundances')
        self.non_symmetrical_decoder = tf.keras.layers.Conv2D(self.parameters["number_bands"],
                                                              kernel_size=7,
                                                              strides=self.parameters['stride_4'],
                                                              padding=self.parameters['padding_4'],
                                                              activation='linear',
                                                              kernel_constraint=tf.keras.constraints.non_neg(),
                                                              kernel_initializer=tf.keras.initializers.RandomNormal(0.0,
                                                                                                                    0.3,
                                                                                                                    seed=123),
                                                              use_bias=False)

    def call(self, data):
        architecture_decoder = self.non_symmetrical_decoder(data)
        return architecture_decoder

    def abundances_estimation(self):
        abundances = self.non_symmetrical_decoder.get_weights()[0]
        return abundances
