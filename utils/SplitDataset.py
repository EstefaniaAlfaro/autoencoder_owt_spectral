import random
import numpy as np
import math


class SplitData:
    def __init__(self, batch_size_data, training_percentage, testing_percentage, validation_percentage):
        self.batch_size_data = batch_size_data
        self.training_percentage = training_percentage
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage

    def split_percentage(self):
        dataset_length = self.batch_size_data.shape[0]
        training_quantity = round((dataset_length * self.training_percentage) / 100)
        testing_quantity = math.ceil(abs(dataset_length * self.testing_percentage) / 100)
        validation_quantity = round(abs(dataset_length * self.validation_percentage) / 100)
        return training_quantity, testing_quantity, validation_quantity, dataset_length

    def mapping_split_data(self, samples_number, random_generation):
        batch_width = self.batch_size_data.shape[1]
        batch_height = self.batch_size_data.shape[-1]
        split_dataset = np.zeros((samples_number, batch_width, batch_height))
        for index in range(samples_number):
            split_dataset[index, :, :] = self.batch_size_data[random_generation[index], :, :]
        return split_dataset

    def split_dataset(self):
        training_quantity, testing_quantity, validation_quantity, dataset_length = self.split_percentage()
        np.random.seed(1)
        random_generation = np.random.randint(dataset_length, size=dataset_length)
        training_random = random_generation[:training_quantity]
        testing_random = random_generation[training_quantity:training_quantity + 1 + testing_quantity]
        validation_random = random_generation[training_quantity + testing_quantity + 1:]
        training_data = self.mapping_split_data(training_random.shape[0], training_random)
        testing_data = self.mapping_split_data(testing_random.shape[0], testing_random)
        validation_data = self.mapping_split_data(validation_random.shape[0], validation_random)
        return training_data, testing_data, validation_data, (training_random, testing_random, validation_random)
