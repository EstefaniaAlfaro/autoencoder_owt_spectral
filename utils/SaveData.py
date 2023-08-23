from scipy.io import savemat
import os

EXTENSION = '.mat'


class SaveDataPreprocess:
    def __init__(self, path: str, title: str, **kwargs):
        self.path = path
        self.title = title
        self.data = kwargs['data']
        self.label = kwargs['label']

    def save_data(self, dictionary):
        results_path = os.path.join(self.path, self.title)
        savemat(results_path + EXTENSION, dictionary)

    def save_dictionary_data(self):
        dictionary = dict()
        aux = 0
        for index in self.label:
            dictionary[index] = self.data[aux]
            aux = aux + 1
        self.save_data(dictionary)
        return dictionary



