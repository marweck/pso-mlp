from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class SwarmConfig:
    number_of_particles: int
    particle_size: int
    lower_bound: float
    upper_bound: float
