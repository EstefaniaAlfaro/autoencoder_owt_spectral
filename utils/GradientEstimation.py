import tensorflow as tf
import keras
from CustomLossFunctions import *


class CustomGradientModel(keras.Model):
    # def __init__(self):
    #     super(CustomGradientModel, self).__init__()
    #     self.model = model

    # def compile(self, optimizer, loss):
    #     super(CustomGradientModel, self).compile()
    #     self.optimizer = optimizer
    #     self.loss = loss

    def train_gradient_step(self, data):
        data_x, data_y = data
        with tf.GradientTape() as tape:
            data_predicted = self(data_x, training=True)
            # loss_function = self.loss(data_y, data_predicted, regularization_losses=self.losses)
            loss_function = loss_function_spectral_angle(data_predicted, data_y)
            # loss_function = self.compiled_loss(data_y, data_predicted, regularization_losses=self.losses)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss_function, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.compiled_metrics.update_state(data_y, data_predicted)
        return {index.name: index.result() for index in self.metrics} \


    # def train_gradient_step(self, data):
    #     with tf.GradientTape() as tape:
    #         data_predicted = self.model(data, training=True)
    #         loss_function = self.loss(data, data_predicted)
    #     trainable_variables = self.trainable_variables
    #     gradients = tape.gradient(loss_function, trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    #     self.compiled_metrics.update_state(data, data_predicted)
    #     return {index.name: index.result() for index in self.metrics}
