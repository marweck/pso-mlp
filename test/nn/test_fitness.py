import unittest

import numpy as np

from nn.fitness import MlpFitness
from nn.mlp import MLP


class MlpFitnessTestCase(unittest.TestCase):
    def test_mlp_fitness_creation(self):
        mlp = MLP((3, 7, 2))

        x_input = np.random.uniform(size=(6, 3))
        y_output = np.random.uniform(size=(6, 2))

        fitness = MlpFitness(mlp, x_input, y_output)

        self.assertIs(fitness.__dict__['_MlpFitness__mlp'], mlp)
        self.assertIs(fitness.__dict__['_MlpFitness__x_input'], x_input)
        self.assertIs(fitness.__dict__['_MlpFitness__y_output'], y_output)

    def test_mlp_fitness_evaluation(self):
        mlp = MLP((3, 7, 2))

        x_input = np.random.uniform(size=(6, 3))
        y_output = np.random.uniform(size=(6, 2))

        fitness = MlpFitness(mlp, x_input, y_output)
        evaluation = fitness.evaluate()
        print('MLP fitness: ', evaluation)

        self.assertTrue(evaluation > 0)
