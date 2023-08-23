import tensorflow as tf
from tensorflow.keras import layers


class ExtractPatches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def get_configurations(self):
        configurations = super().get_configurations().copy()
        configurations.update({
                "input_shape": input_shape,
                "patch_size": patch_size,
                "number_patches": number_patches,
                "projection_dim": projection_dim,
                "number_heads": number_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "units": units,
            })
        return configurations

    def call(self, data):
        batch_size = tf.shape(data)[0]
        patches = tf.image.extract_patches(
            images=data,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        dataset_reshape = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
        return dataset_reshape

