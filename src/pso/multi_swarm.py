import sys
from typing import Dict

import numpy as np

from pso.strategy import OptimizationStrategy, MultiParticle
from pso.swarm import Swarm, SwarmConfig


class MultiSwarm:
    def __init__(self, outer_swarm_config: SwarmConfig, strategy: OptimizationStrategy):
        self.__strategy = strategy
        self.__number_of_particles = outer_swarm_config.number_of_particles
        self.__main_swarm = Swarm(outer_swarm_config)
        self.__inner_progress = []

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
            inner_swarm.add_best_particle(
                self.__best_inner_position_map[i].inner_position.copy(),
                self.__best_inner_position_map[i].fitness
            )

            print('## flying inner swarm for {} iterations'.format(inner_iterations))
            inner_swarm.fly(inner_iterations, evaluator)

            self.__record_best_inner_position(
                i, position_i, inner_swarm.best_swarm_fitness(), inner_swarm.best_position()
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
            outer_position=outer_position.copy(),
            inner_position=best_inner_position.copy(),
        )
        self.__inner_progress.append(best_fitness)

    def __update_best_inner_positions(self):
        for i in range(self.__number_of_particles):
            position_i = self.__main_swarm.particle_position(i)
            best_position_so_far = self.__strategy.best_inner_position_for_outer_particle(
                position_i, self.__best_inner_position_map[i]
            )
            self.__best_inner_position_map[i].inner_position = best_position_so_far.copy()

    def best_outer_position(self) -> np.ndarray:
        return self.__main_swarm.best_position()

    def best_outer_fitness(self) -> float:
        return self.__main_swarm.best_swarm_fitness()

    def best_inner_position(self) -> np.ndarray:
        index = self.__main_swarm.best_particle_index()
        return self.__best_inner_position_map[index].inner_position

    def best_inner_fitness(self) -> float:
        index = self.__main_swarm.best_particle_index()
        return self.__best_inner_position_map[index].fitness

    def best_multi_particle(self) -> MultiParticle:
        index = self.__main_swarm.best_particle_index()
        return self.__best_inner_position_map[index]

    def inner_swarm_fitness_progress(self):
        return self.__inner_progress

    def outer_swarm_fitness_progress(self):
        return self.__main_swarm.fitness_progress()
