from typing import Callable, List

import numpy as np
import sys

from pso.config import SwarmConfig
from pso.speed import SpeedCalculator
from util.random import random_position


class Swarm:
    def __init__(self, config: SwarmConfig,
                 speed_calculator: SpeedCalculator = SpeedCalculator(),
                 create_position: Callable = random_position,
                 create_speed: Callable = random_position):
        self.__swarm_config = config
        self.__speed_calculator = speed_calculator
        self.__best_particle_index = 0

        self.__swarm_position = create_position(config.number_of_particles, config.particle_size,
                                                config.lower_bound, config.upper_bound)
        self.__speed = create_speed(config.number_of_particles, config.particle_size,
                                    -config.upper_bound, config.upper_bound)

        self.__best_position = np.zeros((config.number_of_particles, config.particle_size))
        self.__best_fitness = np.repeat(float(sys.maxsize), config.number_of_particles)
        self.__progress_track = []

    def fly(self, iterations: int, evaluate_swarm: Callable[[int, np.ndarray], float]):
        for i in range(iterations):
            self.__calculate_bests(evaluate_swarm)
            self.__update_swarm()
            self.__save_progress()

    def __calculate_bests(self, evaluate_swarm: Callable[[int, np.ndarray], float]):
        fitness = self.__current_fitness(evaluate_swarm)

        for i in range(self.__swarm_config.number_of_particles):
            if fitness[i] < self.__best_fitness[i]:
                self.__best_fitness[i] = fitness[i]
                self.__best_position[i] = self.__swarm_position[i]

                if fitness[i] < self.best_swarm_fitness():
                    self.__best_particle_index = i

    def __current_fitness(self, evaluate_swarm: Callable[[int, np.ndarray], float]) -> np.ndarray:
        fitness_list = []
        for i in range(self.__swarm_config.number_of_particles):
            fitness_list.append(evaluate_swarm(i, self.__swarm_position[i]))
        return np.asarray(fitness_list)

    def __update_swarm(self):
        self.__speed_calculator.start_update(self.best_swarm_fitness())

        for i in range(self.__swarm_config.number_of_particles):
            self.__speed[i] = self.__speed_calculator.calculate_speed(
                config=self.__swarm_config,
                speed=self.__speed[i],
                position=self.__swarm_position[i],
                best_particle_position=self.__best_position[i],
                best_position=self.best_position(),
                particle_index=i,
                best_particle_index=self.best_particle_index()
            )

            self.__swarm_position[i] = np.clip(self.__swarm_position[i] + self.__speed[i],
                                               a_min=self.__swarm_config.lower_bound,
                                               a_max=self.__swarm_config.upper_bound)

    def __save_progress(self):
        self.__progress_track.append(self.best_swarm_fitness())

    def add_best_particle(self, particle: np.ndarray, fitness: float = float(sys.maxsize)):
        if not np.all(self.__swarm_config.lower_bound <= particle) and \
               np.all(particle <= self.__swarm_config.upper_bound):
            raise ValueError(
                'All values of the new particle should be {:.5f} <= x <= {:.5f}'.
                format(self.__swarm_config.lower_bound, self.__swarm_config.upper_bound)
            )

        self.__swarm_position = np.vstack([self.__swarm_position, particle.copy()])

        speed = random_position(1, self.__swarm_config.particle_size, -self.__swarm_config.upper_bound,
                                self.__swarm_config.upper_bound)
        self.__speed = np.vstack([self.__speed, speed])

        self.__best_position = np.vstack([self.__best_position, particle.copy()])
        self.__best_fitness = np.hstack([self.__best_fitness, fitness])

        self.__best_particle_index = self.__swarm_config.number_of_particles
        self.__swarm_config.number_of_particles += 1

    def particle_position(self, particle_index: int) -> np.ndarray:
        return self.__swarm_position[particle_index]

    def best_swarm_fitness(self) -> float:
        return self.__best_fitness[self.__best_particle_index]

    def fitness_progress(self) -> List[float]:
        return self.__progress_track

    def best_position(self) -> np.ndarray:
        return self.__swarm_position[self.__best_particle_index]

    def best_particle_index(self) -> int:
        return self.__best_particle_index

    def config(self) -> SwarmConfig:
        return self.__swarm_config
