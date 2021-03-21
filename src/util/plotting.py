from typing import List

import matplotlib.pyplot as plt


def plot_fitness(
        inner_swarm_fitness_progress: List[float],
        outer_swarm_fitness_progress: List[float]
):
    plt.plot(inner_swarm_fitness_progress, label='Inner Swarms')
    plt.plot(outer_swarm_fitness_progress, label='Outer Swarm')
    plt.legend()
    plt.ylabel('Fitness')
    plt.title('MultiSwarm Fitness')
    plt.show()
