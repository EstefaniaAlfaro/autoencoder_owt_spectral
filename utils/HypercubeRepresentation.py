import numpy as np


class HypercubeDataRepresentation:
    def __init__(self, dataset):
        self.dataset = dataset

    def reshape_hypercube(self, width, height):
        bands_number = self.dataset.shape[0]
        hypercube_reshape = np.zeros((width, height, bands_number))
        for index in range(bands_number):
            hypercube_reshape[:, :, index] = np.reshape(self.dataset[index, :], (width, height))
        return hypercube_reshape

