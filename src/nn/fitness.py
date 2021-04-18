import numpy as np
from sklearn import metrics

from nn.mlp import MLP


class MlpFitness:

    def __init__(self, mlp: MLP, x_input: np.ndarray, y_output: np.ndarray):
        self.__mlp = mlp
        self.__x_input = np.atleast_2d(x_input)
        self.__y_output = np.atleast_2d(y_output)

    def evaluate(self) -> float:
        y_predicted = np.apply_along_axis(self.__mlp.run, 1, self.__x_input)
        return metrics.mean_squared_error(self.__y_output, y_predicted)
