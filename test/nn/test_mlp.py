import unittest

import numpy as np

from nn.mlp import mlp_shape_dimension, weights_to_layers, create_mlp_layers, MLP, resize_2d_matrix


class MLPTestCase(unittest.TestCase):

    def test_mlp_in_use(self):
        network = MLP((4, 5, 2))
        y = network.run(np.random.uniform(size=4))
        print(y)

        self.assertIsNotNone(y)

        network = MLP((5, 6, 4, 2))
        y = network.run(np.random.uniform(size=5))
        print(y)

        self.assertIsNotNone(y)

    def test_mlp_shape_dimension(self):
        self.assertEqual(37, mlp_shape_dimension((4, 5, 2)))
        self.assertEqual(74, mlp_shape_dimension((5, 6, 4, 2)))

    def test_weights_to_layers(self):
        shape = (4, 5, 2)
        weights = np.random.uniform(size=37)

        layers = weights_to_layers(shape, weights)

        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0].shape, (5, 5))
        self.assertEqual(layers[1].shape, (2, 6))

        shape = (5, 6, 4, 2)
        weights = np.random.uniform(size=74)

        layers = weights_to_layers(shape, weights)

        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0].shape, (6, 6))
        self.assertEqual(layers[1].shape, (4, 7))
        self.assertEqual(layers[2].shape, (2, 5))

    def test_create_mlp_layers(self):
        layers = create_mlp_layers((4, 5, 2))
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0].shape, (5, 5))
        self.assertEqual(layers[1].shape, (2, 6))

        layers = create_mlp_layers((5, 6, 4, 2))
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0].shape, (6, 6))
        self.assertEqual(layers[1].shape, (4, 7))
        self.assertEqual(layers[2].shape, (2, 5))

    def test_get_weights(self):
        shape = (4, 5, 2)
        weights = np.random.uniform(size=37)
        mlp = MLP(shape, weights)

        mlp_weights = mlp.weights()

        self.assertTrue(np.all(mlp_weights == weights))

    def test_mlp_resize_preconditions(self):
        shape = (4, 5, 2)
        mlp = MLP(shape)

        with self.assertRaises(ValueError):
            mlp.resize((4, 5, 4, 2))

        with self.assertRaises(ValueError):
            mlp.resize((5, 5, 2))

        with self.assertRaises(ValueError):
            mlp.resize((4, 5, 3))

        self.assertEqual(len(mlp.weights()), 37)

        mlp.resize((4, 7, 2))
        self.assertEqual(len(mlp.weights()), 51)

        mlp.resize((4, 3, 2))
        self.assertEqual(len(mlp.weights()), 23)

    def test_matrix_resize(self):
        matrix = np.random.uniform(size=(5, 4))
        same_matrix = resize_2d_matrix(matrix, (5, 4))
        self.assertIs(matrix, same_matrix)

        smaller = resize_2d_matrix(matrix, (4, 3))
        self.assertEqual(smaller.shape, (4, 3))

        bigger = resize_2d_matrix(matrix, (6, 7))
        self.assertEqual(bigger.shape, (6, 7))

        mixed = resize_2d_matrix(matrix, (4, 7))
        self.assertEqual(mixed.shape, (4, 7))

        mixed2 = resize_2d_matrix(matrix, (5, 2))
        self.assertEqual(mixed2.shape, (5, 2))
