import numpy as np
import statistics


class EnhancementInitializer:
    def __init__(self, patches_selection):
        self.patches_selection = patches_selection

    def reshape_data_patches(self):
        reshape_patches = np.reshape(self.patches_selection[0, :, :], -1)
        return reshape_patches

    def enhancement_energy(self):
        reshape_patches = self.reshape_data_patches()
        energy_enhancement = sum(reshape_patches ** 2)
        return energy_enhancement

    def enhancement_mean(self):
        reshape_patches = self.reshape_data_patches()
        mean_enhancement = np.mean(reshape_patches)
        return mean_enhancement

    def enhancement_standard_deviation(self):
        reshape_patches = self.reshape_data_patches()
        standard_deviation = statistics.stdev(reshape_patches)
        return standard_deviation



