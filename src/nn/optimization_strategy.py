from typing import Callable

import numpy as np

from pso.multi_swarm import OptimizationStrategy
from pso.swarm import Swarm, SwarmConfig


class MlpStrategy(OptimizationStrategy):
    def __init__(self, number_of_outer_particles: int, inner_swarm_config: SwarmConfig):
        self.__inner_config = inner_swarm_config

    def best_inner_position_for_outer_particle(self, index: int, outer_position: np.ndarray,
                                               best_so_far_inner_position: np.ndarray) -> np.ndarray:
        # Need first to adapt best inner position so far to the new outer position
        pass

    def create_inner_swarm(self, outer_position: np.ndarray) -> Swarm:
        return Swarm(self.__inner_config)

    def inner_swarm_evaluator(self, outer_swarm_position_i: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        # creates MLP evaluator function using training dataset
        pass

    def evaluate_outer_swarm(self, position_matrix: np.ndarray) -> np.ndarray:
        # MLP evaluator using validation dataset
        pass
