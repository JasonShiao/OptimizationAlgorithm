from random import random
from math import inf
import numpy as np

def generate_random_solutions(num_solutions, lower_bounds, upper_bounds):
    # Create a Generator using default BitGenerator
    generator = np.random.default_rng()

    # Generate random solutions
    random_solutions = generator.uniform(lower_bounds, upper_bounds, size=(num_solutions, len(lower_bounds)))
    return random_solutions

class Particle:
    def __init__(self, pos, v):
        self.pos = pos
        self.v = v
        self.p_best = self.pos # initialize with current pos
        self.best_value = inf
    
    def __str__(self):
        return f"pos: {self.pos}, v: {self.v}, p_best: {self.p_best}, best_val: {self.best_value}"

class PSO:
    """
        Particle Swarm Optimization
    """    
    n_dim = 0
    particles = [] # A list of Particle
    g_best = {'idx': 0, 'pos': [], 'value': inf}
    def __init__(self, acc_coeff_1, acc_coeff_2, num_particle, weight):
        self.acc_coeff_1 = acc_coeff_1 # individual acceleration coefficient, typically 2
        self.acc_coeff_2 = acc_coeff_2 # social acceleration coefficient, typically 2
        self.num_particle = num_particle
        self.weight = weight # inertia weight, typically 0.5 - 0.9, (default: 0.8 for general purpose)
    def update_particles(self, bounds_min, bounds_max, v_max):
        for idx, particle in enumerate(self.particles):
            particle.v = [self.weight * particle.v[d] + self.acc_coeff_1 * random() * (particle.p_best[d] - particle.pos[d]) + self.acc_coeff_2 * random() * (self.g_best['pos'][d] - particle.pos[d]) 
                            for d in range(self.n_dim)]
            # Restrict velocity [-v_max, v_max]
            particle.v = [ v_max[d] if particle.v[d] > v_max[d] else -v_max[d] if particle.v[d] < -v_max[d] else particle.v[d]
                            for d in range(self.n_dim)]
            particle.pos = [particle.pos[d] + particle.v[d] 
                            for d in range(self.n_dim)]
            # Apply Bounce rule
            for d in range(self.n_dim):
                if particle.pos[d] > bounds_max[d]:
                    particle.pos[d] = bounds_max[d]
                    particle.v[d] = -particle.v[d]
                elif particle.pos[d] < bounds_min[d]:
                    particle.pos[d] = bounds_min[d]
                    particle.v[d] = -particle.v[d]
            self.particles[idx] = particle
    
    def optimize(self, objective_func, lb, ub, n_dim, epoch, tolerance, verbose = False):
        """_summary_

        Args:
            objective_func (_type_): _description_
            lb (_type_): _description_
            ub (_type_): _description_
            n_dim (_type_): _description_
            epoch (_type_): _description_
            tolerance (_type_): The acceptable score to stop the optimization
            verbose (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # Validate and adjust input
        if len(lb) == 1:
            lb = lb * n_dim # Extend to n_dim
        if len(ub) == 1:
            ub = ub * n_dim # Extend to n_dim
        if len(lb) != n_dim or len(ub) != n_dim:
            raise ValueError(f"Invalid lb or ub size, expected {n_dim} but got {len(lb)} and {len(ub)}")
        # Initialize parameters and particles
        self.n_dim = n_dim
        self.g_best = {'idx': 0, 'pos': [], 'value': inf}
        # Generate v_max from boundary
        #v_max = [ (ub[d] - lb[d]) / (epoch / 5) for d in range(n_dim)]
        v_max = list((np.array(ub) - np.array(lb)) / 2)
        # Generate random initial_pos and initial_v based on bounds
        init_pos = generate_random_solutions(self.num_particle, lb, ub)
        init_v = generate_random_solutions(self.num_particle, list(-np.array(v_max)), v_max)
        self.particles = [Particle(pos, vel) for pos, vel in zip(init_pos, init_v)]
        
        # Main PSO loop
        iter = 0
        while True: 
            # 1. Evaluate each particle, update p_best and g_best
            for idx in range(self.num_particle):
                fitness_value = objective_func(self.particles[idx].pos)
                # Check and Update particle best
                if fitness_value < self.particles[idx].best_value:
                    self.particles[idx].best_value = fitness_value
                    self.particles[idx].p_best = self.particles[idx].pos
                    # Check and Update global best
                    if fitness_value < self.g_best['value']:
                        self.g_best['value'] = fitness_value
                        self.g_best['pos'] = self.particles[idx].pos
                        self.g_best['idx'] = idx
            # 2. Update particle
            self.update_particles(lb, ub, v_max)
            # 3. Reach terminate condition?
            iter += 1
            # Print state
            if verbose:
                print(f"epoch = {iter}, g_best = {self.g_best}")
            # Terminate condition
            if iter >= epoch or self.g_best['value'] < tolerance:
                break
        return self.particles, self.g_best
