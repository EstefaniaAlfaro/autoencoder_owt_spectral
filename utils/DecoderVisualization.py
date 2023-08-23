from matplotlib import pyplot as plt
import tensorflow as tf


class DecoderVisualization:
    def __init__(self, testing_data, reconstructed_data):
        self.testing_data = testing_data
        self.reconstructed_data = reconstructed_data

    def decoded_images(self, number_images):
        for index in range(number_images):
            ax = plt.subplot(2, number_images, index + 1)
            if len(self.testing_data.shape) == 4:
                plt.imshow(tf.squeeze(tf.transpose(self.testing_data[index, :, :])))
            else:
                plt.imshow(tf.transpose(self.testing_data[index, :, :]))
            plt.title('Original', fontsize=5)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(2, number_images, index + 1 + number_images)
            if len(self.reconstructed_data.shape) == 4:
                plt.imshow(tf.squeeze(self.reconstructed_data[index, :, :].T))
            else:
                plt.imshow(self.reconstructed_data[index, :, :].T)
            plt.title('Reconstructed', fontsize=5)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
