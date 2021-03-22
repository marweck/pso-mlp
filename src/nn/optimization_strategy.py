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

    def initial_inner_position_for_outer_position(self, outer_position: np.ndarray) -> np.ndarray:
        number_of_weights = self.__inner_swarm_dimension_for_outer_particle(outer_position)
        return np.random.uniform(size=number_of_weights,
                                 low=self.inner_config.lower_bound,
                                 high=self.inner_config.upper_bound)

    def best_inner_position_for_outer_particle(self, outer_position: np.ndarray,
                                               best_so_far: MultiParticle) -> np.ndarray:
        shape = shape_from_outer_position(best_so_far.outer_position, self.number_of_inputs, self.number_of_outputs)
        weights = best_so_far.inner_position.copy()
        mlp = MLP(shape=shape, weights=weights)

        # resize mlp to the shape of outer_position
        new_shape = shape_from_outer_position(outer_position, self.number_of_inputs, self.number_of_outputs)
        mlp.resize(new_shape)

        return mlp.weights()

    def create_inner_swarm(self, outer_position: np.ndarray) -> Swarm:
        number_of_weights = self.__inner_swarm_dimension_for_outer_particle(outer_position)
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
        shape = shape_from_outer_position(outer_swarm_position_i, self.number_of_inputs, self.number_of_outputs)

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
            shape = shape_from_outer_position(best_positions[index].outer_position,
                                              self.number_of_inputs, self.number_of_outputs)
            weights = best_positions[index].inner_position

            mlp = MLP(shape=shape, weights=weights)

            # resize mlp to new shape
            new_shape = shape_from_outer_position(position_i, self.number_of_inputs, self.number_of_outputs)
            mlp.resize(new_shape)

            fitness = MlpFitness(mlp, x_input=self.x_validation, y_output=self.y_validation)
            return fitness.evaluate()

        return evaluate

    def __inner_swarm_dimension_for_outer_particle(self, outer_position: np.ndarray) -> int:
        shape = shape_from_outer_position(outer_position, self.number_of_inputs, self.number_of_outputs)
        return mlp_shape_dimension(shape)


def shape_from_outer_position(outer_position: np.ndarray,
                              number_of_inputs: int,
                              number_of_outputs: int) -> Tuple[int, ...]:
    shape = [int(x) for x in outer_position]
    shape.insert(0, number_of_inputs)
    shape.append(number_of_outputs)
    return tuple(shape)
