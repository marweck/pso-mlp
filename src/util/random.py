import numpy as np


def random_position(number_of_particles: int, size: int,
                    lower_bound: float, upper_bound: float) -> np.ndarray:
    return np.random.uniform(size=(number_of_particles, size), low=lower_bound, high=upper_bound)
