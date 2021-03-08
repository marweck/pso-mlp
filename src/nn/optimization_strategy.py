from typing import Callable, Dict

import numpy as np

from pso.multi_swarm import OptimizationStrategy, MultiParticle
from pso.swarm import Swarm, SwarmConfig


class MlpStrategy(OptimizationStrategy):
    def __init__(self, inner_swarm_config: SwarmConfig,
                 x_training: np.ndarray,
                 y_training: np.ndarray,
                 x_validation: np.ndarray,
                 y_validation: np.ndarray):
        self.__inner_config = inner_swarm_config
        self.x_training = x_training
        self.y_training = y_training
        self.x_validation = x_validation
        self.y_validation = y_validation

    def best_inner_position_for_outer_particle(self, index: int, outer_position: np.ndarray,
                                               best_so_far_inner_position: np.ndarray) -> np.ndarray:
        # Need first to adapt best inner position so far to the new outer position
        pass

    def create_inner_swarm(self, outer_position: np.ndarray) -> Swarm:
        return Swarm(self.__inner_config)

    def inner_swarm_evaluator(self, outer_swarm_position_i: np.ndarray) -> \
            Callable[[int, np.ndarray], np.ndarray]:
        # creates MLP evaluator function using training dataset
        pass

    def outer_swarm_evaluator(self, best_positions: Dict[int, MultiParticle]) -> \
            Callable[[int, np.ndarray], float]:
        # MLP evaluator using validation dataset
        pass
