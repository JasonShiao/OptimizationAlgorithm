from random import random
from math import inf

class PSO:
    """
        Particle Swarm Optimization
    """    
    n_dim = 0
    particles = [] # A list of Particle
    g_best = {'idx': 0, 'pos': [], 'value': inf}
    def __init__(self, acc_coeff_1, acc_coeff_2, num_particle, weight):
        self.acc_coeff_1 = acc_coeff_1
        self.acc_coeff_2 = acc_coeff_2
        self.num_particle = num_particle
        self.weight = weight
    def update_particles(self, bound_min, bound_max):
        for idx, particle in enumerate(self.particles):
            particle.v = [self.weight * particle.v[d] + self.acc_coeff_1 * random() * (particle.p_best[d] - particle.pos[d]) + self.acc_coeff_2 * random() * (self.g_best['pos'][d] - particle.pos[d]) 
                            for d in range(self.n_dim)]
            # TODO: Restrict velocity
            ........
            particle.pos = [particle.pos[d] + particle.v[d] 
                            for d in range(self.n_dim)]
            # Bounce
            for d in range(self.n_dim):
                if particle.pos[d] > bound_max[d]:
                    particle.pos[d] = bound_max[d]
                    particle.v[d] = -particle.v[d]
                elif particle.pos[d] < bound_min[d]:
                    particle.pos[d] = bound_min[d]
                    particle.v[d] = -particle.v[d]
            self.particles[idx] = particle
    
    def optimize(self, objective_func, n_dim, max_iter, tolerance, init_pos = None, init_v = None):
        self.n_dim = n_dim
        self.g_best = {'idx': 0, 'pos': [], 'value': inf}
        if init_pos and init_v:
            self.particles = [self.Particle(init_pos[i], init_v[i]) for i in range(self.num_particle)]
        else:
            # TODO: Randomly initialize particles
            return None
        # Main PSO loop
        iter = 0
        while True: 
            # 1. Evaluate each particle, update p_best and g_best
            for idx in range(self.num_particle):
                fitness_value = objective_func(self.particles[idx].pos)
                # Check particle best
                if fitness_value < self.particles[idx].best_value:
                    self.particles[idx].best_value = fitness_value
                    self.particles[idx].p_best = self.particles[idx].pos
                    # Check global best
                    if fitness_value < self.g_best['value']:
                        self.g_best['value'] = fitness_value
                        self.g_best['pos'] = self.particles[idx].pos
                        self.g_best['idx'] = idx
            # 2. Update particle
            self.update_particles(*(objective_func.suggested_bounds()))
            # 3. Reach terminate condition?
            iter += 1
            if iter >= max_iter or self.g_best['value'] < tolerance:
                break
        return self.particles, self.g_best
    
    
    class Particle:
        def __init__(self, pos, v):
            self.pos = pos
            self.v = v
            self.p_best = self.pos # initialize with current pos
            self.best_value = inf
        
        def __str__(self):
            return f"pos: {self.pos}, v: {self.v}, p_best: {self.p_best}, best_val: {self.best_value}"