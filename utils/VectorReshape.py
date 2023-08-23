import tensorflow as tf


class ImageReshapeTensor:
    def __init__(self, data, batch_size, number_rows, number_columns, depth):
        self.data = data
        self.batch_size = batch_size
        self.number_rows = number_rows
        self.number_columns = number_columns
        self.depth = depth

    def reshape_tensor_representation(self):
        image_tensor_representation = tf.expand_dims(self.data, 0)
        return image_tensor_representation

    def batch_size_selection(self):
        data_tensor_representation = self.reshape_tensor_representation()
        extracted_patches = tf.image.extract_patches(data_tensor_representation,
                                                     sizes=[1, self.batch_size,  self.batch_size, 1],
                                                     strides=[1, self.batch_size, self.batch_size, 1],
                                                     rates=[1, 1, 1, 1],
                                                     padding='VALID')
        return extracted_patches


