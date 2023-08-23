from matplotlib import pyplot as plt
# import matplotlib as mpl
# import numpy as np
# mpl.use('TkAgg')


class VisualizationResults:
    def __init__(self, end_members_dictionary: dict, abundances_dictionary: dict):
        self.end_members_dictionary = end_members_dictionary
        self.abundances_dictionary = abundances_dictionary

    def end_members_visualization(self):
        remove_headers_end_members_dictionary = dict(list(self.end_members_dictionary.items())[3:])
        for key, value in remove_headers_end_members_dictionary.items():
            plt.plot(value[:, 0], linewidth=1.5)
            plt.plot(value[:, 1], linewidth=1.5)
            plt.title(key)
            plt.grid()
            plt.show()
            plt.figure()

    def abundances_visualization(self, image_width, image_height):
        remove_headers_abundances_dictionary = dict(list(self.abundances_dictionary.items())[3:])
        for key, value in remove_headers_abundances_dictionary.items():
            abundances_maps_reshape_ground_truth = np.reshape(value[:, 0], (image_width, image_height))
            abundances_maps_reshape_prediction = np.reshape(value[:, 1], (image_width, image_height))
            plt.imshow(abundances_maps_reshape_ground_truth)
            plt.show()
            plt.figure()
            plt.imshow(abundances_maps_reshape_prediction)
            plt.show()