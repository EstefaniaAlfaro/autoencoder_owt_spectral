from sklearn import preprocessing


class ScalingDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def normalization_min_max(self):
        min_max_scaling = preprocessing.MinMaxScaler()
        min_max_scaling_normalization = min_max_scaling.fit_transform(self.dataset)
        return min_max_scaling_normalization

    def normalization_max_abs(self):
        max_abs_scaling = preprocessing.MaxAbsScaler()
        max_abs_scaling_normalization = max_abs_scaling.fit_transform(self.dataset)
        return max_abs_scaling_normalization
