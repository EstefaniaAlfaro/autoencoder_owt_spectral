from sklearn.feature_extraction.image import extract_patches_2d


class TrainingPatchesData:
    def __init__(self, hypercube, patch_size, patch_number, batch_size):
        self.hypercube = hypercube
        self.patch_size = patch_size
        self.patch_number = patch_number
        self.batch_size = batch_size

    def data_patches(self):
        extract_patches_data = extract_patches_2d(self.hypercube, (self.patch_size, self.patch_size))
        return extract_patches_data

