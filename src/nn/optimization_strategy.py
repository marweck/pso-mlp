from typing import Callable, Dict, List

import numpy as np

from nn.fitness import MlpFitness
from nn.mlp import MLP
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

        self.number_of_inputs = len(x_training[0])
        self.number_of_outputs = len(y_training[0])

    def best_inner_position_for_outer_particle(self, index: int, outer_position: np.ndarray,
                                               best_so_far_inner_position: np.ndarray) -> np.ndarray:
        # Need first to adapt best inner position so far to the new outer position
        pass

    def create_inner_swarm(self, outer_position: np.ndarray) -> Swarm:
        return Swarm(self.__inner_config)

    def inner_swarm_evaluator(self, outer_swarm_position_i: np.ndarray) -> \
            Callable[[int, np.ndarray], float]:
        """
        Creates MLP evaluator function using training dataset
        """
        shape = [round(x) for x in outer_swarm_position_i]
        shape.insert(0, self.number_of_inputs)
        shape.append(self.number_of_outputs)

        def evaluate(index: int, position_i: np.ndarray) -> float:
            mlp = MLP(shape=tuple(shape), weights=position_i)
            fitness = MlpFitness(mlp, x_input=self.x_training, y_output=self.y_training)
            return fitness.evaluate()

        return evaluate

    def outer_swarm_evaluator(self, best_positions: Dict[int, MultiParticle]) -> \
            Callable[[int, np.ndarray], float]:
        """
        Creates MLP evaluator function using validation dataset
        """
        def evaluate(index: int, position_i: np.ndarray) -> float:
            pass

        return evaluate
