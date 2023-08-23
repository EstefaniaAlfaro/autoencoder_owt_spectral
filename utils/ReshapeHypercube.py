import tensorflow as tf
import pandas as pd


class ReshapeHypercube:
    def __init__(self, hypercube_dataset):
        self.hypercube_dataset = hypercube_dataset

    def reshape_hypercube_dataframe(self):
        transpose_batch_size = tf.transpose(self.hypercube_dataset).numpy()
        reshape_hypercube = transpose_batch_size.reshape(-1, transpose_batch_size.shape[-1])
        hypercube_dataframe = pd.DataFrame(data=reshape_hypercube)
        hypercube_dataframe.columns = [f'abundances-seg{index}' for index in range(1, 1 +
                                                                                   transpose_batch_size.shape[-1])]
        return hypercube_dataframe
