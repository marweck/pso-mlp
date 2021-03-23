from typing import Dict, Callable

import numpy as np

from pso.swarm import Swarm


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
