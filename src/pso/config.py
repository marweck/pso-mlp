class SwarmConfig:
    def __init__(self, number_of_particles: int, size: int, lower_bound: float, upper_bound: float):
        self.number_of_particles = number_of_particles
        self.particle_size = size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
