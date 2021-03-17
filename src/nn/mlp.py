from typing import Tuple, List
import numpy as np

from scipy import special


class MLP:
    def __init__(self, shape: Tuple[int, ...], weights: np.ndarray = None):
        self.__shape = shape
        self.__number_of_layers = len(shape) - 1

        if weights is None:
            self.__layers = create_mlp_layers(shape)
        else:
            self.__layers = weights_to_layers(shape, weights)

    def run(self, input_data: np.ndarray) -> np.ndarray:
        layer = input_data.T

        for i in range(self.__number_of_layers):
            previous_layer = np.insert(layer, 0, 1, axis=0)
            output = np.dot(self.__layers[i], previous_layer)

            # logistic sigmoid
            layer = special.expit(output)

        return layer

    def resize(self, new_shape: Tuple[int, ...]):
        if len(new_shape) != len(self.__shape):
            raise ValueError('New shape must have {0} layers'.format(len(self.__shape)))

        if new_shape[0] != self.__shape[0] or \
                new_shape[self.__number_of_layers] != self.__shape[self.__number_of_layers]:
            raise ValueError('New shape must have the same number of inputs and outputs')

        for i in range(self.__number_of_layers):
            self.__layers[i] = resize_2d_matrix(self.__layers[i], (new_shape[i + 1], new_shape[i] + 1))

    def weights(self) -> np.ndarray:
        weights = np.asarray([])
        for i in range(self.__number_of_layers):
            weights = np.hstack((weights, np.concatenate(self.__layers[i])))
        return weights


def create_mlp_layers(shape: Tuple[int, ...]) -> List[np.ndarray]:
    layers = []
    num_layers = len(shape) - 1

    for i in range(num_layers):
        layer = np.random.uniform(size=(shape[i + 1], shape[i] + 1))
        layers.append(layer)

    return layers


def weights_to_layers(shape: Tuple[int, ...], weights: np.ndarray) -> List[np.ndarray]:
    dimension = mlp_shape_dimension(shape)
    weights_dimension = len(weights)
    if weights_dimension != dimension:
        raise ValueError('The weights vector for shape {} should have length {}, but has length {}'
                         .format(shape, dimension, weights_dimension))

    num_layers = len(shape) - 1
    layers = []
    offset = 0

    for i in range(num_layers):
        layer_length = shape[i + 1] * (shape[i] + 1)
        vector = weights[offset:offset + layer_length]
        layer = np.reshape(vector, (shape[i + 1], shape[i] + 1))

        layers.append(layer)

        offset += layer_length

    return layers


def mlp_shape_dimension(shape: Tuple[int, ...]) -> int:
    return np.sum([shape[i + 1] * (shape[i] + 1) for i in range(len(shape) - 1)])


def resize_2d_matrix(matrix: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
    if len(matrix.shape) != 2:
        raise ValueError('Matrix should be a 2d matrix')

    shape = matrix.shape

    diff_rows = new_shape[0] - shape[0]
    diff_cols = new_shape[1] - shape[1]

    if diff_rows > 0:
        matrix = np.row_stack((matrix, np.random.uniform(size=(diff_rows, shape[1]))))
    elif diff_rows < 0:
        matrix = matrix[:diff_rows, :]

    if diff_cols > 0:
        matrix = np.column_stack((matrix, np.random.uniform(size=(new_shape[0], diff_cols))))
    elif diff_cols < 0:
        matrix = matrix[:, :diff_cols]

    return matrix
