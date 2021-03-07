import unittest
from unittest.mock import patch

import sys
import numpy as np
from pso.swarm import Swarm, SwarmConfig


class SwarmTestCase(unittest.TestCase):

    @patch('pso.swarm.np.random.uniform')
    def test_create_swarm(self, uniform_patch):
        uniform_patch.return_value = np.zeros((20, 50))
        config = SwarmConfig(number_of_particles=20, size=50, lower_bound=-0.5, upper_bound=0.5)
        swarm = Swarm(config)

        best_fitness = swarm.best_swarm_fitness()
        self.assertEqual(sys.maxsize, best_fitness)

        self.assertTrue(np.all(swarm.best_position() == 0))

    def test_swarm_convergence(self):
        config = SwarmConfig(number_of_particles=20, size=10, lower_bound=-1, upper_bound=1)
        swarm = Swarm(config)
        swarm.add_particle(np.random.uniform(size=10, low=-1, high=1))

        solution = np.random.uniform(size=10)
        fitness = Fitness(solution)
        print(solution)
        previous_best = sys.maxsize

        for i in range(10):
            swarm.fly(3, fitness.evaluate)
            print(swarm.best_position())
            best_swarm_fitness = swarm.best_swarm_fitness()
            print('best fitness so far: {:.5f}'.format(best_swarm_fitness))

            self.assertTrue(best_swarm_fitness <= previous_best)
            previous_best = best_swarm_fitness

        best_index = swarm.best_particle_index()
        self.assertTrue(np.all(swarm.best_position() == swarm.particle_position(best_index)))

    def test_non_negative_particle_swarm(self):
        config = SwarmConfig(number_of_particles=20, size=6, lower_bound=1, upper_bound=16)
        swarm = Swarm(config)
        solution = np.random.uniform(size=6, low=5, high=9)
        fitness = Fitness(solution)
        print(solution)

        for i in range(3):
            swarm.fly(10, fitness.evaluate)
            print(swarm.best_position())
            print('best fitness so far: {:.5f}'.format(swarm.best_swarm_fitness()))

        best = swarm.best_position()
        self.assertTrue(np.all(best >= 1) and np.all(best <= 16))

    def test_add_particle(self):
        config = SwarmConfig(number_of_particles=5, size=6, lower_bound=1, upper_bound=10)
        swarm = Swarm(config)

        swarm.add_particle(np.ones(6))

        self.assertTrue(np.all(swarm.particle_position(5) == 1.))

        with self.assertRaises(ValueError):
            swarm.add_particle(np.zeros(6))


class Fitness:
    def __init__(self, solution: np.ndarray):
        self.__solution = solution

    def evaluate(self, position_matrix: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.__distance, 1, position_matrix)

    def __distance(self, row):
        """
        Euclidean distance
        """
        return np.linalg.norm(self.__solution - row)
