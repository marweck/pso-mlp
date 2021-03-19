import sys
from typing import Callable, Dict

import numpy as np

from pso.swarm import Swarm, SwarmConfig


class MultiParticle:
    def __init__(self, fitness: float, inner_position: np.ndarray, outer_position: np.ndarray):
        self.fitness = fitness
        self.inner_position = inner_position
        self.outer_position = outer_position


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

    def initial_inner_position_for_outer_position(self, outer_position: np.ndarray) -> np.ndarray:
        """
        Determines the initial position of an inner particle based on an outer particle position
        :param outer_position:
        """
        pass

    def best_inner_position_for_outer_particle(self, outer_position: np.ndarray,
                                               best_so_far: MultiParticle) -> np.ndarray:
        """
        Returns the best inner position for outer particle "index". This position keeps track of the best
        inner position visited so far in an outer PSO round. Its dimension should vary overtime to reflect
        the changes from outer particle i position.

        This particle position will be passed over to the next round of the inner PSO
        :param outer_position: current outer particle index position vector
        :param best_so_far: best known MultiParticle position from an inner swarm for outer particle 'index'
        :return: best inner position recorded for outer particle index adapted to current outer particle
                 position
        """
        pass

    def inner_swarm_evaluator(self, outer_swarm_position_i: np.ndarray) \
            -> Callable[[int, np.ndarray], float]:
        """
        Receives a particle position from the outer swarm and creates a fitness function to be used on the
        inner PSO

        :param outer_swarm_position_i: particle i position to be used on inner PSO
        :return: fitness function to be used on inner PSO
        """
        pass

    def outer_swarm_evaluator(self, best_positions: Dict[int, MultiParticle]) \
            -> Callable[[int, np.ndarray], float]:
        """
        Fitness function for the outer PSO

        :param best_positions: dictionary of MultiParticle by outer particle index
        :return: vector of fitness values for each outer particle position
        """
        pass


class MultiSwarm:
    def __init__(self, outer_swarm_config: SwarmConfig, strategy: OptimizationStrategy):
        self.__strategy = strategy
        self.__number_of_particles = outer_swarm_config.number_of_particles
        self.__main_swarm = Swarm(outer_swarm_config)

        self.__best_inner_position_map: Dict[int, MultiParticle] = {
            i: MultiParticle(
                fitness=sys.maxsize,
                inner_position=strategy.initial_inner_position_for_outer_position(
                    self.__main_swarm.particle_position(i)
                ),
                outer_position=self.__main_swarm.particle_position(i)
            )
            for i in range(self.__number_of_particles)
        }

    def fly(self, iterations: int, inner_iterations: int):
        for i in range(iterations):
            print('# outer iteration ', i + 1)
            self.__fly_inner_swarms(inner_iterations)
            self.__fly_outer_swarm()

    def __fly_inner_swarms(self, inner_iterations: int):
        for i in range(self.__number_of_particles):
            position_i = self.__main_swarm.particle_position(i)

            inner_swarm = self.__strategy.create_inner_swarm(position_i)
            evaluator = self.__strategy.inner_swarm_evaluator(position_i)

            # adds best inner position so far to keep memory of previous good solutions
            inner_swarm.add_particle(self.__best_inner_position_map[i].inner_position.copy())

            print('## flying inner swarm for {} iterations'.format(inner_iterations))
            inner_swarm.fly(inner_iterations, evaluator)

            self.__record_best_inner_position(
                i, position_i.copy(), inner_swarm.best_swarm_fitness(), inner_swarm.best_position()
            )

    def __fly_outer_swarm(self):
        outer_evaluator = self.__strategy.outer_swarm_evaluator(self.__best_inner_position_map)
        self.__main_swarm.fly(1, outer_evaluator)
        self.__update_best_inner_positions()

    def __record_best_inner_position(self, outer_index: int, outer_position: np.ndarray,
                                     best_fitness: float, best_inner_position: np.ndarray):

        print('## best fitness for inner particle {} = {}'.format(outer_index, best_fitness))
        self.__best_inner_position_map[outer_index] = MultiParticle(
            fitness=best_fitness,
            outer_position=outer_position,
            inner_position=best_inner_position,
        )

    def __update_best_inner_positions(self):
        for i in range(self.__number_of_particles):
            position_i = self.__main_swarm.particle_position(i)
            best_position_so_far = self.__strategy.best_inner_position_for_outer_particle(
                position_i, self.__best_inner_position_map[i]
            )
            self.__best_inner_position_map[i].inner_position = best_position_so_far

    def best_outer_position(self) -> np.ndarray:
        return self.__main_swarm.best_position()

    def best_outer_fitness(self) -> float:
        return self.__main_swarm.best_swarm_fitness()

    def best_inner_position(self) -> np.ndarray:
        index = self.__main_swarm.best_particle_index()
        return self.__best_inner_position_map[index].inner_position

    def best_multi_particle(self) -> MultiParticle:
        index = self.__main_swarm.best_particle_index()
        return self.__best_inner_position_map[index]
