import unittest

import numpy as np

from nn.mlp import mlp_shape_dimension
from nn.optimization_strategy import MlpStrategy
from pso.swarm import SwarmConfig


class MlpStrategyTestCase(unittest.TestCase):

    def test_mlp_strategy_creation(self):
        swarm_config = SwarmConfig(number_of_particles=20, size=30, lower_bound=-0.5, upper_bound=0.5)
        x_training = np.random.uniform(size=(20, 7))
        y_training = np.random.uniform(size=(20, 2))
        x_validation = np.random.uniform(size=(10, 7))
        y_validation = np.random.uniform(size=(10, 2))

        strategy = MlpStrategy(
            inner_swarm_config=swarm_config,
            x_training=x_training,
            y_training=y_training,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        self.assertIs(strategy.inner_config, swarm_config)
        self.assertIs(strategy.x_training, x_training)
        self.assertIs(strategy.y_training, y_training)
        self.assertIs(strategy.x_validation, x_validation)
        self.assertIs(strategy.y_validation, y_validation)
        self.assertIs(strategy.number_of_inputs, 7)
        self.assertIs(strategy.number_of_outputs, 2)

    def test_create_inner_swarm(self):
        swarm_config = SwarmConfig(number_of_particles=20, size=30, lower_bound=-0.5, upper_bound=0.5)
        x_training = np.random.uniform(size=(20, 7))
        y_training = np.random.uniform(size=(20, 2))
        x_validation = np.random.uniform(size=(10, 7))
        y_validation = np.random.uniform(size=(10, 2))

        strategy = MlpStrategy(
            inner_swarm_config=swarm_config,
            x_training=x_training,
            y_training=y_training,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        dimension = mlp_shape_dimension((7, 3, 5, 2))

        swarm = strategy.create_inner_swarm(np.asarray([3.4, 5.6]))
        self.assertEqual(swarm.config().upper_bound, swarm_config.upper_bound)
        self.assertEqual(swarm.config().lower_bound, swarm_config.lower_bound)
        self.assertEqual(swarm.config().number_of_particles, swarm_config.number_of_particles)
        self.assertEqual(swarm.config().particle_size, dimension)

    def test_best_inner_position_for_outer_particle(self):
        pass  # TODO test best inner position transformation

    def test_inner_swarm_evaluator(self):
        pass  # TODO test inner swarm evaluator factory

    def test_outer_swarm_evaluator(self):
        pass  # TODO test outer swarm evaluator factory
