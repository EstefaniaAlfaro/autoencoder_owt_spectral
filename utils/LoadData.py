import scipy.io


class LoadDataset:
    def __init__(self, image_path, ground_truth_path, data_target, ground_truth_target, ground_truth_abundances):
        self.image_path = image_path
        self.ground_truth_path = ground_truth_path
        self.data_target = data_target
        self.ground_truth_target = ground_truth_target
        self.ground_truth_abundances = ground_truth_abundances

    def load_data_matlab_extension(self):
        load_data = scipy.io.loadmat(self.image_path)
        load_ground_truth = scipy.io.loadmat(self.ground_truth_path)
        return load_data, load_ground_truth

    def data_dictionary_extraction(self):
        load_data, load_ground_truth = self.load_data_matlab_extension()
        dataset = load_data[self.data_target]
        ground_truth = load_ground_truth[self.ground_truth_target]
        ground_truth_abundances = load_ground_truth[self.ground_truth_abundances]
        bands_number = dataset.shape[0]
        return dataset, ground_truth, ground_truth_abundances, bands_number

    def data_number_images(self):
        data, ground_truth, ground_truth_abundances, bands_number = self.data_dictionary_extraction()
        return data, ground_truth, ground_truth_abundances, bands_number



