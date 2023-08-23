import json
import os.path

EXTENSION = '.json'

'''
Usage parameters 
filename = {
    "patch_size_model": 40,
    "batch_size": 40,
    "data_preparation_title": 'data_preparation_' + str(40),
    "filters_l1": 48,
    "kernel_size_l1": 3,
    "padding_l1": "same",
    "strides_l1": 1,
    "filters_l2": 4,
    "kernel_size_l2": 3,
    "padding_l2": "same",
    "strides_l2": 1,
    "filters_decoder": 198,
    "kernel_size_decoder": 13,
    "padding_decoder": "same",
    "strides_decoder": 1,
}
for save the json file:

SaveConfigurationFile(directory, 'title_configurations').save_configurations_file(filename)
'''


class SaveConfigurationFile:
    def __init__(self, directory: str, configurations_file: str):
        self.directory = directory
        self.configurations_file = configurations_file

    def folder_configurations(self):
        full_path = os.path.join(self.directory, self.configurations_file)
        return full_path

    def save_configurations_file(self, filename):
        path = self.folder_configurations()
        configuration_json_file = json.dumps(filename)
        json_filename = open(path + EXTENSION, 'w')
        json_filename.write(configuration_json_file)
        json_filename.close()

    def load_configurations_file(self):
        full_path = self.folder_configurations()
        load_json_filename = open(full_path + EXTENSION, 'r')
        data_configuration = json.loads(load_json_filename.read())
        load_json_filename.close()
        return data_configuration
