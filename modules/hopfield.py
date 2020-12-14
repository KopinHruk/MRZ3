"""Module with Hopfield network"""
import glob
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from modules.dataset import Dataset


class HopfieldNetwork:
    """Hopfiled network implementation"""

    def __init__(self, train_data: List[np.ndarray],
                 verbose: bool = True):
        """
        Init methods

        :param train_data: data for network training
        :param verbose: if True shows progress bar
        """

        self.train_data = train_data
        self.num_neurons = train_data[0].shape[0]

        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        self.verbose = verbose

    def train(self):
        """Trains models using examples from train_data"""

        copied_train_data = np.copy(self.train_data)
        number_of_images = len(copied_train_data)

        for curr_copied_sample in tqdm(copied_train_data,
                                       disable=not self.verbose,
                                       postfix=f'Training...'):
            temp1 = curr_copied_sample.reshape(-1, 1) @ np.linalg.pinv(curr_copied_sample.reshape(-1, 1))
            self.weights += np.sign(temp1)

        # diagonal_values = np.diag(self.weights)  # extracts diagonal values from matrix
        # diagonal_weights = np.diag(diagonal_values)  # creates diagonal matrix from diagonal values for weights
        #
        # self.weights = self.weights - diagonal_weights
        self.weights = self.weights / number_of_images

    def predict(self, data: List[np.ndarray], num_iter: int = 40) -> List[np.ndarray]:
        """
        Predicts data class

        :param data: list of ndarrays with data
        :param num_iter: number of iterations for image restoring
        :param threshold: threshold for sign ( sign(x - threshold) )
        :return: resulted data
        """

        copied_data = np.copy(data)

        predicted_data = list()

        for curr_copied_sample in tqdm(copied_data,
                                       disable=not self.verbose,
                                       postfix=f'Predicting...'):
            curr_prediction = np.sign(self.weights.dot(curr_copied_sample))
            predicted_data.append(curr_prediction)

        return predicted_data



if __name__ == '__main__':
    # data = [np.random.random((15, 15)).flatten() for i in range(1)]
    image_paths = glob.glob(pathname='../../images_same/*.*', recursive=True)
    image_size = (64, 64)

    dataset = Dataset(list_of_paths=image_paths, image_size=image_size)
    flatten_images = dataset.get_all_flatten_images() # dataset.get_all_flatten_images() #[dataset.get_all_flatten_images()[0]]

    test_dataset = Dataset(list_of_paths=image_paths, image_size=image_size, add_noise=True)
    test_images = [test_dataset.get_all_flatten_images()[1]]# test_dataset.get_all_flatten_images() #[test_dataset.get_all_flatten_images()[0]]

    net = HopfieldNetwork(train_data=flatten_images)

    net.train()
    pred = net.predict(test_images, 40, 50)

    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(np.array(test_images).reshape(64, 64))
    axs[1].imshow(np.array(pred).reshape(64, 64))
    axs[2].imshow(net.weights)

    plt.show()
