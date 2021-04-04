import sys

import numpy as np

from pso.config import SwarmConfig

# constriction factor
CHI = 0.7298437881283576

# acceleration constants
C1 = 2.05
C2 = 2.05

# convergence factors
MINIMUM_RHO = 1e-30
INCREASE_FACTOR = 1.5
DECREASE_FACTOR = 0.5


class SpeedCalculator:
    """
    This class implements the vanilla PSO speed calculation
    """

    def start_update(self, best_fitness: float):
        pass

    def calculate_speed(self,
                        config: SwarmConfig,
                        speed: np.ndarray,
                        position: np.ndarray,
                        best_particle_position: np.ndarray,
                        best_position: np.ndarray,
                        particle_index: int,
                        best_particle_index: int):

        r1 = np.random.uniform(size=config.particle_size)
        r2 = np.random.uniform(size=config.particle_size)

        momentum = CHI * speed
        personal_term = C1 * r1 * (best_particle_position - position)
        global_term = C2 * r2 * (best_position - position)

        return np.clip(momentum + personal_term + global_term,
                       a_min=-config.upper_bound,
                       a_max=config.upper_bound)


class ConvergingSpeedCalculator(SpeedCalculator):
    """
    This class implements the Guaranteed Convergence PSO (GCPSO) speed calculation
    GC PSO was introduced by the article 'A Guaranteed Global Convergence Particle Swarm Optimizer'
    https://link.springer.com/chapter/10.1007/978-3-540-25929-9_96
    """
    def __init__(self):
        super().__init__()

        self.__rho = 1.0
        self.__successes = 0
        self.__failures = 0
        self.__failure_threshold = 15
        self.__success_threshold = 5
        self.__old_best_fitness = sys.maxsize

    def start_update(self, best_fitness: float):
        if best_fitness != self.__old_best_fitness:
            self.__successes += 1
            self.__failures = 0
        else:
            self.__successes = 0
            self.__failures += 1

        if self.__successes > self.__success_threshold:
            self.__rho *= INCREASE_FACTOR
            self.__success_threshold += 1
        elif self.__failures > self.__failure_threshold:
            self.__rho *= DECREASE_FACTOR
            self.__failure_threshold += 1

        if self.__rho < MINIMUM_RHO:
            self.__rho = MINIMUM_RHO

        self.__old_best_fitness = best_fitness

    def calculate_speed(self,
                        config: SwarmConfig,
                        speed: np.ndarray,
                        position: np.ndarray,
                        best_particle_position: np.ndarray,
                        best_position: np.ndarray,
                        particle_index: int,
                        best_particle_index: int):

        if particle_index != best_particle_index:
            return super().calculate_speed(
                config=config,
                speed=speed,
                position=position,
                best_particle_position=best_particle_position,
                best_position=best_position,
                particle_index=particle_index,
                best_particle_index=best_particle_index
            )
        else:
            r1 = np.random.uniform(size=config.particle_size)
            return best_particle_position - position + CHI * speed + self.__rho * (1 - 2 * r1)
