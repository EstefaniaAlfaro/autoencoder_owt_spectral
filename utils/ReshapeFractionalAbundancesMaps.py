import numpy as np


class ReshapeFractionalMaps:
    def __init__(self, abundances_maps, image_width, image_height, parameters):
        self.abundances_maps = abundances_maps
        self.image_width = image_width
        self.image_height = image_height
        self.parameters = parameters

    def reshape_abundances_maps(self):
        abundances_maps = np.zeros((self.image_width, self.image_height, self.parameters["number_end_members"]))
        for index in range(self.parameters["number_end_members"]):
            abundances_maps[:, :, index] = np.reshape(self.abundances_maps[index], (self.image_width, self.image_height))
        return abundances_maps
