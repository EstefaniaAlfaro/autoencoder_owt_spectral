import os
from scipy.io import savemat

PATH_SAVE = '../results'
EXTENSION = '.mat'


class SaveDictionaryFormat:
    def __init__(self, dictionary_data, title: str,):
        self.dictionary_data = dictionary_data
        self.title = title

    def save_dictionary_format(self):
        full_path = os.path.join(PATH_SAVE, self.title)
        try:
            savemat(full_path + EXTENSION, self.dictionary_data)
        except NameError:
            print('Is not possible to write the file')

