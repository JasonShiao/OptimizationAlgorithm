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
    temp = 0
    history = []
    
    def __init__(self, init_temp, terminate_temp, cooling_rate, max_iter, gen_func):
        self.init_temp = init_temp
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
    
    def gaussian_gen_func(self, x, bound_min, bound_max):
        """
            Generate random solution with multivariate Gaussian distribution
                mean: current solution: x
                sigma: temperature -> covariate matrix (for multivariate gaussian)
        """
        current_x = np.array(x)
        cov = np.identity(current_x.size) * self.temp
        next_x = np.random.multivariate_normal(x, cov, size=1)[0]
        # Apply constraint (bounds)
        next_x = np.clip(next_x, a_min = np.array(bound_min), a_max = np.array(bound_max))
        return next_x.tolist()

    def cauchy_gen_func(self, x, bound_min, bound_max):
        pass
    
    def check_accept(self, delta_e):
        """
            Determine whether accept the new solution
        """
        return (delta_e <= 0 or 
                exp(-delta_e / self.temp) > random())
    
    def annealing(self):
        # Annealing type 1: cool by scale
        self.temp *= self.cooling_rate
        # Annealing type 2: cool by constant temp
        # self.temp -= self.cooling_rate
    
    def optimize(self, objective_func, n_dim, initial_solution = None):
        """_summary_

        Args:
            objective_func (_type_): any fitness function class object with the same API as classes in benchmark_functions
            n_dim (_type_): number of dimension for solution
            initial_solution (_type_, optional): Initial solution for optimization. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.iter = 1
        self.temp = self.init_temp
        self.solution = None
        self.energy = inf
        # 1. Initialize a random solution
        min_bounds, max_bounds = objective_func.suggested_bounds()
        if not initial_solution:
            new_solution = [random() * (max_bounds[idx] - min_bounds[idx]) + min_bounds[idx] for idx in range(n_dim)]
        else:
            new_solution = initial_solution
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
                    self.iter = 1
                    # TODO: Next step with generating function
                    new_solution = self.gen_func(self.solution, min_bounds, max_bounds)
            else:
                self.iter += 1
                # TODO: Next step with generating function
                new_solution = self.gen_func(self.solution, min_bounds, max_bounds)
        return self.solution, self.energy
        
