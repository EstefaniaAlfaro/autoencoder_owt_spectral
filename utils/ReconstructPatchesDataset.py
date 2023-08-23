from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import numpy as np


class ReconstructTrainingPatches:
    def __init__(self, patches, image_width, image_height, number_bands):
        self.patches = patches
        self.image_width = image_width
        self.image_height = image_height
        self.number_bands = number_bands

    def reconstruct_data_patches(self):
        if (np.squeeze(self.patches).shape[0], np.squeeze(self.patches).shape[1]) \
                != (self.image_width, self.image_height):
            reconstructed_data = reconstruct_from_patches_2d(self.patches, (self.image_width,
                                                                            self.image_height, self.number_bands))
        else:
            reconstructed_data = self.patches
        return reconstructed_data
