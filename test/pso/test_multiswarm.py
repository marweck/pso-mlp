import unittest
from typing import Callable

import numpy as np
import sys

from pso.multi_swarm import MultiSwarm, OptimizationStrategy
from pso.swarm import Swarm, SwarmConfig


class MultiSwarmTestCase(unittest.TestCase):

    def test_multi_swarm_creation(self):
        strategy = SimpleStrategy()
        config = SwarmConfig(number_of_particles=5, size=1, lower_bound=1, upper_bound=5)
        multi_swarm = MultiSwarm(outer_swarm_config=config, strategy=strategy)

        self.assertTrue(multi_swarm.best_outer_fitness() == sys.maxsize)
        self.assertTrue(np.all(multi_swarm.best_outer_position() >= 1))
        self.assertTrue(np.all(multi_swarm.best_outer_position() <= 5))
        self.assertTrue(np.all(multi_swarm.best_inner_position() == 0.))

    def test_multi_swarm_convergence(self):
        strategy = SimpleStrategy()
        config = SwarmConfig(number_of_particles=5, size=1, lower_bound=1, upper_bound=5)
        multi_swarm = MultiSwarm(outer_swarm_config=config, strategy=strategy)

        previous_best = sys.maxsize

        for i in range(5):
            multi_swarm.fly(iterations=1, inner_iterations=7)
            best_swarm_fitness = multi_swarm.best_outer_fitness()
            print('best fitness so far: {:.5f}'.format(best_swarm_fitness))

            self.assertTrue(best_swarm_fitness <= previous_best)
            previous_best = best_swarm_fitness

        self.assertTrue(np.all(multi_swarm.best_outer_position() >= 1))
        self.assertTrue(np.all(multi_swarm.best_outer_position() <= 5))
        self.assertTrue(np.all(multi_swarm.best_inner_position() >= -1))
        self.assertTrue(np.all(multi_swarm.best_inner_position() <= 1))


class SimpleStrategy(OptimizationStrategy):
    def __init__(self):
        self.__target = np.random.uniform(size=6, low=1, high=5)
        self.__inner_target = np.random.uniform(size=12, low=-1, high=1)

    def best_inner_position_for_outer_particle(self, index: int, outer_position: np.ndarray,
                                               best_so_far_inner_position: np.ndarray) -> np.ndarray:
        print('index: ', index)
        print('outer position: ', outer_position)
        if len(best_so_far_inner_position) < 12:
            return np.zeros(12)
        return best_so_far_inner_position

    def create_inner_swarm(self, outer_position: np.ndarray) -> Swarm:
        config = SwarmConfig(number_of_particles=15, size=12, lower_bound=-1, upper_bound=1)
        return Swarm(config)

    def evaluate_outer_swarm(self, position_matrix: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self.__outer_distance, 1, position_matrix)

    def inner_swarm_evaluator(self, outer_swarm_position_i: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        return lambda matrix: np.apply_along_axis(self.__inner_distance, 1, matrix)

    def __outer_distance(self, row):
        return np.linalg.norm(self.__target - row)

    def __inner_distance(self, row):
        return np.linalg.norm(self.__inner_target - row)
