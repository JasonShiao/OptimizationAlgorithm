from math import exp, inf
from random import random
from enum import Enum
import numpy as np
import benchmark_functions as bf

class NextStepPolicy(Enum):
    Gaussian = 'gaussian'
    Cauchy = 'cauchy'
    
class CoolingSchedule(Enum):
    Scale = 'scale'
    Constant = 'constant'

class SA:
    """ 
        Simulated Annealing algorithm
    """
    solution = None
    iter = 1
    energy = inf # initialize with positive infinite
    temp = 0
    history = []
    
    def __init__(self, init_temp: float, terminate_temp: float, cooling_sched: CoolingSchedule, cooling_rate: float,
                 max_epoch: int, max_iter_per_temp: int, next_step_policy = NextStepPolicy.Cauchy):
        self.init_temp = init_temp
        self.terminate_temp = terminate_temp
        if init_temp < terminate_temp:
            raise ValueError("Initial temp must be higher than terminate temperature")
        self.cooling_sched = cooling_sched
        self.cooling_rate = cooling_rate
        self.max_epoch = max_epoch
        self.max_iter_per_temp = max_iter_per_temp # Max iteration for "a temperature"
        if next_step_policy in [NextStepPolicy.Gaussian, NextStepPolicy.Cauchy]:
            self.next_step_policy = next_step_policy
        else:
            raise ValueError("gen_func must be 'gaussian' or 'cauchy'")
    
    def next_step(self, current_sol, bound_min, bound_max):
        """
            Generate random solution with multivariate Gaussian distribution
                mean: current solution: x
                sigma: temperature -> covariate matrix (for multivariate gaussian)
            
            Gaussian (Normal) distribution has a lower peak and thinner tails than Cauchy distribution
            which makes it more prone to be trapped in local optima
            
            Cauchy-Lorentzian distribution has a higher peak and thicker tails than Gaussian distribution
            Thus, it is more likely to trigger a long jump and exit the local optima
        """
        if self.next_step_policy == NextStepPolicy.Gaussian:
            current_x = np.array(current_sol)
            cov = np.identity(current_x.size) * self.temp
            next_x = np.random.multivariate_normal(current_sol, cov, size=1)[0]
            # Apply constraint (bounds)
            next_x = np.clip(next_x, a_min = np.array(bound_min), a_max = np.array(bound_max))
            return next_x.tolist()
        elif self.next_step_policy == NextStepPolicy.Cauchy:
            cauchy_rand = np.random.standard_cauchy(len(current_sol))
            next_x = cauchy_rand * self.temp + np.array(current_sol)
            # Apply constraint (bounds)
            next_x = np.clip(next_x, a_min = np.array(bound_min), a_max = np.array(bound_max))
            return next_x.tolist()
        else:
            raise ValueError("Unknown next step policy")
    
    def check_accept(self, delta_e):
        """
            Determine whether accept the new solution:
            delta_e <= 0: means new solution is better -> always accept
            delta_e > 0: means new solution is worse -> accept with probability exp(-delta_e / temp)
        """
        return (delta_e <= 0 or random() < exp(-delta_e / self.temp))
    
    def annealing(self):
        if self.cooling_sched == CoolingSchedule.Scale:
            # Annealing type 1: cool by scale
            self.temp *= self.cooling_rate
        elif self.cooling_sched == CoolingSchedule.Constant:
            # Annealing type 2: cool by constant temp
            self.temp -= self.cooling_rate
        else:
            raise ValueError("Unknown cooling schedule")        
    
    def optimize(self, objective_func: bf.BenchmarkFunction, n_dim, verbose = False):
        """_summary_

        Args:
            objective_func (_type_): any fitness function class object with the same API as classes in benchmark_functions
            n_dim (_type_): number of dimension for solution
            initial_solution (_type_, optional): Initial solution for optimization. Defaults to None.

        Returns:
            _type_: _description_
        """
        # 1. Initialize a random solution
        min_bounds, max_bounds = objective_func.suggested_bounds()
        epoch = 0
        best_so_far_solution = None
        best_so_far_energy = np.inf
        while epoch < self.max_epoch:
            self.iter = 1
            self.temp = self.init_temp
            self.solution = None
            self.energy = inf
            new_solution = list(np.random.uniform(low=min_bounds, high=max_bounds, size=n_dim))
            while True:
                # 2. Evaluate
                new_energy = objective_func(new_solution)
                if new_energy == None:
                    raise ValueError("Objective function must return a value")
                delta_e = new_energy - self.energy
                if self.check_accept(delta_e):
                    if new_energy < best_so_far_energy:
                        best_so_far_energy = new_energy
                        best_so_far_solution = new_solution
                    self.solution = new_solution
                    self.energy = new_energy
                # 3. Check reach max iter for current temperature
                if self.iter >= self.max_iter_per_temp:
                    # If reach terminate temperature -> terminate
                    if self.temp < self.terminate_temp:
                        break
                    else:
                        self.annealing()
                        self.iter = 1
                        # TODO: Next step with generating function
                        new_solution = self.next_step(self.solution, min_bounds, max_bounds)
                else:
                    self.iter += 1
                    # TODO: Next step with generating function
                    new_solution = self.next_step(self.solution, min_bounds, max_bounds)
            epoch += 1
            if verbose:
                print(f"Epoch {epoch} best solution = {best_so_far_solution}, best energy = {best_so_far_energy}")
        return best_so_far_solution, best_so_far_energy
