import unittest

import numpy as np

from nn.mlp import mlp_shape_dimension
from nn.optimization_strategy import MlpStrategy
from pso.multi_swarm import MultiSwarm, MultiParticle
from pso.swarm import SwarmConfig
from util.plotting import plot_fitness


class MlpStrategyTestCase(unittest.TestCase):

    def test_mlp_strategy_creation(self):
        swarm_config = SwarmConfig(number_of_particles=20, particle_size=30, lower_bound=-0.5, upper_bound=0.5)
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
        swarm_config = SwarmConfig(number_of_particles=20, particle_size=30, lower_bound=-0.5, upper_bound=0.5)
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

    def test_integrated_optimization(self):
        # initial particle size is whatever because it will vary with the architecture
        inner_config = SwarmConfig(number_of_particles=10, particle_size=30, lower_bound=-0.5, upper_bound=0.5)
        x_training = np.random.uniform(size=(20, 7))
        y_training = np.random.uniform(size=(20, 2))
        x_validation = np.random.uniform(size=(10, 7))
        y_validation = np.random.uniform(size=(10, 2))

        strategy = MlpStrategy(
            inner_swarm_config=inner_config,
            x_training=x_training,
            y_training=y_training,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        outer_config = SwarmConfig(number_of_particles=10, particle_size=3, lower_bound=1, upper_bound=12)

        swarm = MultiSwarm(outer_config, strategy)

        swarm.fly(2, 10)
        first_outer_fitness = swarm.best_outer_fitness()
        first_inner_fitness = swarm.best_inner_fitness()

        swarm.fly(3, 10)
        second_outer_fitness = swarm.best_outer_fitness()
        second_inner_fitness = swarm.best_inner_fitness()

        self.assertTrue(second_outer_fitness <= first_outer_fitness)
        self.assertTrue(second_inner_fitness <= first_inner_fitness)

        # plot_fitness(
        #     inner_swarm_fitness_progress=swarm.inner_swarm_fitness_progress(),
        #     outer_swarm_fitness_progress=swarm.outer_swarm_fitness_progress()
        # )

        print('Best architecture: [{}, {}, {}]'.format(7, swarm.best_outer_position(), 2))
        print('Best particle: ', swarm.best_inner_position())
        print('Best outer fitness ', second_outer_fitness)
        print('Best inner fitness ', second_inner_fitness)

    def test_initial_inner_position_for_outer_position(self):
        inner_config = SwarmConfig(number_of_particles=10, particle_size=30, lower_bound=-0.5, upper_bound=0.5)
        x_training = np.random.uniform(size=(20, 7))
        y_training = np.random.uniform(size=(20, 2))
        x_validation = np.random.uniform(size=(10, 7))
        y_validation = np.random.uniform(size=(10, 2))

        strategy = MlpStrategy(
            inner_swarm_config=inner_config,
            x_training=x_training,
            y_training=y_training,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        initial_position = strategy.initial_inner_position_for_outer_position(np.asarray([2, 3]))
        self.assertEqual(len(initial_position), 33)

    def test_best_inner_position_for_outer_particle(self):
        inner_config = SwarmConfig(number_of_particles=10, particle_size=30, lower_bound=-0.5, upper_bound=0.5)
        x_training = np.random.uniform(size=(20, 7))
        y_training = np.random.uniform(size=(20, 2))
        x_validation = np.random.uniform(size=(10, 7))
        y_validation = np.random.uniform(size=(10, 2))

        strategy = MlpStrategy(
            inner_swarm_config=inner_config,
            x_training=x_training,
            y_training=y_training,
            x_validation=x_validation,
            y_validation=y_validation,
        )

        best_so_far = MultiParticle(fitness=0.5,
                                    inner_position=np.random.uniform(size=33),
                                    outer_position=np.asarray([2, 3]))
        inner_position = strategy.best_inner_position_for_outer_particle(
            np.asarray([4, 5]), best_so_far
        )
        self.assertEqual(len(inner_position), 69)
