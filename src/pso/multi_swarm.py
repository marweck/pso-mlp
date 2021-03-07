import sys
from typing import Callable

import numpy as np

from pso.swarm import Swarm, SwarmConfig


class OptimizationStrategy:
    """
    This is an interface that should be implemented by an application specific problem and be used with
    the MultiSwarm class
    """

    def create_inner_swarm(self, outer_position: np.ndarray) -> Swarm:
        """
        Builds a new swarm based on an outer swarm position.

        :param outer_position: position of an outer swarm particle
        :return: Swarm object to be used to evolve inner particles
        """
        pass

    def best_inner_position_for_outer_particle(self, index: int, outer_position: np.ndarray,
                                               best_so_far_inner_position: np.ndarray) -> np.ndarray:
        """
        Returns the best inner position for outer particle "index". This position keeps track of the best
        inner position visited so far in an outer PSO round. Its dimension should vary overtime to reflect
        the changes from outer particle i position.

        This particle position will be passed over to the next round of the inner PSO
        :param index: outer position index
        :param outer_position: current outer particle index position vector
        :param best_so_far_inner_position: best know position for an inner swarm for outer particle index
        :return: best inner position recorded for outer particle index adapted to current outer particle
                 position
        """
        pass

    def evaluate_outer_swarm(self, position_matrix: np.ndarray) -> np.ndarray:
        """
        Fitness function for the outer PSO

        :param position_matrix: current position of all outer particles
        :return: vector of fitness values for each outer particle position
        """
        pass

    def inner_swarm_evaluator(self, outer_swarm_position_i: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """
        Receives a particle position from the outer swarm and creates a fitness function to be used on the inner PSO

        :param outer_swarm_position_i: particle i position to be used on inner PSO
        :return: fitness function to be used on inner PSO
        """
        pass


class MultiSwarm:
    def __init__(self, outer_swarm_config: SwarmConfig,
                 strategy: OptimizationStrategy):
        self.__strategy = strategy
        self.__number_of_particles = outer_swarm_config.number_of_particles
        self.__main_swarm = Swarm(outer_swarm_config)
        self.__best_inner_position_map = {
            i: {
                'fitness': sys.maxsize,
                'position': np.zeros(1),
                'outer_position': None
            } for i in range(self.__number_of_particles)
        }

    def fly(self, iterations: int, inner_iterations: int):
        for i in range(iterations):
            self.__fly_inner_swarms(inner_iterations)
            self.__fly_outer_swarm()

    def __fly_inner_swarms(self, inner_iterations: int):
        for i in range(self.__number_of_particles):
            position_i = self.__main_swarm.particle_position(i)

            inner_swarm = self.__strategy.create_inner_swarm(position_i)
            evaluator = self.__strategy.inner_swarm_evaluator(position_i)

            best_position_so_far = self.__strategy.best_inner_position_for_outer_particle(
                i, position_i, self.__best_inner_position_map[i]['position']
            )
            inner_swarm.add_particle(best_position_so_far)

            inner_swarm.fly(inner_iterations, evaluator)

            self.__record_best_inner_position(
                i, position_i, inner_swarm.best_swarm_fitness(), inner_swarm.best_position()
            )

    def __fly_outer_swarm(self):
        self.__main_swarm.fly(1, self.__strategy.evaluate_outer_swarm)

    def __record_best_inner_position(self, outer_index: int, outer_position: np.ndarray,
                                     best_fitness: np.float64, best_inner_position: np.ndarray):

        if best_fitness < self.__best_inner_position_map[outer_index]['fitness']:
            self.__best_inner_position_map[outer_index] = {
                'fitness': best_fitness,
                'position': best_inner_position,
                'outer_position': outer_position
            }

    def best_outer_position(self) -> np.ndarray:
        return self.__main_swarm.best_position()

    def best_outer_fitness(self) -> np.float64:
        return self.__main_swarm.best_swarm_fitness()

    def best_inner_position(self) -> np.ndarray:
        index = self.__main_swarm.best_particle_index()
        return self.__best_inner_position_map[index]['position']
