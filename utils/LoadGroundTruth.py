import numpy as np


class LoadGroundTruth:
    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height

    def load_abundances_maps_ground_truth(self, abundances_maps):
        number_maps = abundances_maps.shape[0]
        maps_abundances = np.zeros((self.image_height, self.image_width, number_maps))
        for index in range(number_maps):
            maps_abundances[:, :, index] = np.reshape(abundances_maps[index], (self.image_height, self.image_width))
        return maps_abundances
