import tensorflow as tf


def stack_data(patch_image_stack):
    data_stack_patches = tf.concat([patch_image_stack[0], patch_image_stack[1]], axis=-1)
    length_stack = len(patch_image_stack)
    for index in range(length_stack - 2):
        data_stack_patches = tf.concat([data_stack_patches, patch_image_stack[index + 2]], axis=-1)
    transpose_data_stack_patches = tf.transpose(data_stack_patches)
    return transpose_data_stack_patches


class StackPatches:
    def __init__(self, patches, batch_size, batch_size_depth):
        self.patches = patches
        self.batch_size = batch_size
        self.batch_size_depth = batch_size_depth

    def stack_patches_dataset(self):
        patch_image_stack = []
        for index, patch in enumerate(self.patches[0]):
            patch_image_stack.append(tf.reshape(patch, (self.batch_size, self.batch_size,
                                                               self.batch_size_depth)))
        data_stack_patches = stack_data(patch_image_stack)
        return patch_image_stack, data_stack_patches






