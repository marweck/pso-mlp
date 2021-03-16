from typing import Callable, Dict, Tuple

import numpy as np

from nn.fitness import MlpFitness
from nn.mlp import MLP, mlp_shape_dimension
from pso.multi_swarm import OptimizationStrategy, MultiParticle
from pso.swarm import Swarm, SwarmConfig


class MlpStrategy(OptimizationStrategy):
    """
    Strategy for MultiSwarm optimization of MLP neural networks
    """

    def __init__(self, inner_swarm_config: SwarmConfig,
                 x_training: np.ndarray,
                 y_training: np.ndarray,
                 x_validation: np.ndarray,
                 y_validation: np.ndarray):
        self.inner_config = inner_swarm_config

        self.x_training = x_training
        self.y_training = y_training
        self.x_validation = x_validation
        self.y_validation = y_validation

        self.number_of_inputs = len(x_training[0])
        self.number_of_outputs = len(y_training[0])

    def best_inner_position_for_outer_particle(self, index: int, outer_position: np.ndarray,
                                               best_so_far: MultiParticle) -> np.ndarray:
        shape = self.__shape_from_outer_position(best_so_far.outer_position)
        weights = best_so_far.inner_position
        mlp = MLP(shape=tuple(shape), weights=weights)

        # resize mlp to the shape of outer_position
        new_shape = self.__shape_from_outer_position(outer_position)
        mlp.resize(new_shape)

        return mlp.weights()

    def create_inner_swarm(self, outer_position: np.ndarray) -> Swarm:
        shape = self.__shape_from_outer_position(outer_position)
        number_of_weights = mlp_shape_dimension(tuple(shape))
        config = SwarmConfig(
            number_of_particles=self.inner_config.number_of_particles,
            size=number_of_weights,
            lower_bound=self.inner_config.lower_bound,
            upper_bound=self.inner_config.upper_bound
        )
        return Swarm(config)

    def inner_swarm_evaluator(self, outer_swarm_position_i: np.ndarray) -> \
            Callable[[int, np.ndarray], float]:
        """
        Creates MLP evaluator function using training dataset
        """
        shape = self.__shape_from_outer_position(outer_swarm_position_i)

        def evaluate(index: int, position_i: np.ndarray) -> float:
            mlp = MLP(shape=shape, weights=position_i)
            fitness = MlpFitness(mlp, x_input=self.x_training, y_output=self.y_training)
            return fitness.evaluate()

        return evaluate

    def outer_swarm_evaluator(self, best_positions: Dict[int, MultiParticle]) -> \
            Callable[[int, np.ndarray], float]:
        """
        Creates MLP evaluator function using validation dataset
        """
        def evaluate(index: int, position_i: np.ndarray) -> float:
            shape = self.__shape_from_outer_position(best_positions[index].outer_position)
            weights = best_positions[index].inner_position

            mlp = MLP(shape=shape, weights=weights)

            # resize mlp to new shape
            new_shape = self.__shape_from_outer_position(position_i)
            mlp.resize(new_shape)

            fitness = MlpFitness(mlp, x_input=self.x_validation, y_output=self.y_validation)
            return fitness.evaluate()

        return evaluate

    def __shape_from_outer_position(self, outer_position: np.ndarray) -> Tuple[int, ...]:
        shape = [round(x) for x in outer_position]
        shape.insert(0, self.number_of_inputs)
        shape.append(self.number_of_outputs)
        return tuple(shape)
