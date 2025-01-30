import numpy as np
import random
import tensorflow as tf


class PlaceboBase:
    def __init__(self, config):
        self.config = config
        try:
            self.arr_data = np.load(self.config.modelspecific.data_path[0])["arr"]
        except:
            self.arr_data = np.load(self.config.modelspecific.data_path[0])["arr_0"]
        self.create_list_of_data()

    def create_list_of_data(self):
        self.list_data = list(self.arr_data)

    def synthesize_data(self):
        max_index = len(self.list_data) - 1

        sample = None
        if not self.list_data:
            print("### WARNING, DATASET HAS NOT ENOUGH SAMPLES ###")
        else:
            sample = self.list_data.pop(random.randint(0, max_index))

        # print(sample.shape)
        sample = np.expand_dims(sample, 0)
        return tf.convert_to_tensor(sample)

    def build(self, input_shape):
        return

    def summary(self, expand_nested):
        return
