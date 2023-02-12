from math import exp, inf
from random import random, gauss
import numpy as np

class SA:
    """ 
        Simulated Annealing algorithm
    """
    solution = None
    iter = 1
    energy = inf # initialize with positive infinite
    min_energy = inf
    max_energy = -inf
    
    def __init__(self, init_temp, terminate_temp, cooling_rate, max_iter, gen_func):
        self.temp = init_temp
        self.terminate_temp = terminate_temp
        if init_temp < terminate_temp:
            raise ValueError("Initial temp must be higher than terminate temperature")
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter # Max iteration for "a temperature"
        if gen_func == 'gaussian':
            self.gen_func = self.gaussian_gen_func
        elif gen_func =='cauchy':
            self.gen_func = self.cauchy_gen_func
        else:
            raise ValueError("gen_func must be 'gaussian' or 'cauchy'")
    
    def gaussian_gen_func(self, x):
        """
            Generate random solution with multivariate Gaussian distribution
                mean: current solution: x
                sigma: temperature -> covariate matrix (for multivariate gaussian)
        """
        cov = np.identity(x.size) * self.temp
        next_x = np.random.multivariate_normal(x, cov, size=1)[0]
        return next_x

    def cauchy_gen_func(self, x):
        pass
    
    def check_accept(self, delta_e):
        """
            Determine whether accept the new solution
        """
        return (delta_e <= 0 or 
                exp(-delta_e / self.temp) > random())
    
    def annealing(self):
        # cool by scale
        self.temp *= self.cooling_rate
        # cool by constant temp
        # self.temp -= self.cooling_rate
    
    def optimize(self, objective_func, num_var, initial_solution, constraint):
        # 1. Initialize a random solution
        new_solution = initial_solution
        # [random() * (constraint['max'] - constraint['min']) + constraint['min'] for x in range(num_var)]
        self.iter = 1
        while True:
            # 2. Evaluate
            new_energy = objective_func(new_solution)
            delta_e = new_energy - self.energy
            if self.check_accept(delta_e):
                self.solution = new_solution
                self.energy = new_energy
            # 3. Check reach max iter for current temperature
            if self.iter >= self.max_iter:
                # If reach terminate temperature -> terminate
                if self.temp < self.terminate_temp:
                    break
                else:
                    self.annealing()
                    iter = 1
                    # TODO: Next step with generating function
                    new_solution = self.gen_func(self.solution)
            else:
                self.iter += 1
                # TODO: Next step with generating function
                new_solution = self.gen_func(self.solution)
        return self.solution, self.energy
        
