#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Python code created by "Thieu Nguyen" at 21:29, 12/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

# Main paper (Please refer to the main paper):
# Slime Mould Algorithm: A New Method for Stochastic Optimization
# Shimin Li, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, Seyedali Mirjalili
# Future Generation Computer Systems,2020
# DOI: https://doi.org/10.1016/j.future.2020.03.055
# https://www.sciencedirect.com/science/article/pii/S0167739X19320941
# ------------------------------------------------------------------------------------------------------------
# Website of SMA: http://www.alimirjalili.com/SMA.html
# You can find and run the SMA code online at http://www.alimirjalili.com/SMA.html

# You can find the SMA paper at https://doi.org/10.1016/j.future.2020.03.055
# Please follow the paper for related updates in researchgate: 
# https://www.researchgate.net/publication/340431861_Slime_mould_algorithm_A_new_method_for_stochastic_optimization
# ------------------------------------------------------------------------------------------------------------
#  Main idea: Shimin Li
#  Author and programmer: Shimin Li,Ali Asghar Heidari,Huiling Chen
#  e-Mail: simonlishimin@foxmail.com
# ------------------------------------------------------------------------------------------------------------
#  Co-author:
#             Huiling Chen(chenhuiling.jlu@gmail.com)
#             Mingjing Wang(wangmingjing.style@gmail.com)
#             Ali Asghar Heidari(aliasghar68@gmail.com, as_heidari@ut.ac.ir)
#             Seyedali Mirjalili(ali.mirjalili@gmail.com)
#             
#             Researchgate: Ali Asghar Heidari https://www.researchgate.net/profile/Ali_Asghar_Heidari
#             Researchgate: Seyedali Mirjalili https://www.researchgate.net/profile/Seyedali_Mirjalili
#             Researchgate: Huiling Chen https://www.researchgate.net/profile/Huiling_Chen
# ------------------------------------------------------------------------------------------------------------

# More details on the imported benchmark functions:
#   https://gitlab.com/luca.baronti/python_benchmark_functions

from optimization.third_party.slime_mold_algorithm.SMA import BaseSMA, OriginalSMA
from numpy import sum, pi, exp, sqrt, cos
import benchmark_functions as bf


def func_ackley(solution):
    a, b, c = 20, 0.2, 2 * pi
    d = len(solution)
    sum_1 = -a * exp(-b * sqrt(sum(solution ** 2) / d))
    sum_2 = exp(sum(cos(c * solution)) / d)
    return sum_1 - sum_2 + a + exp(1)


def sma_example(args):
    # Determine benchmark function, dimension and bounds
    if args == None or not hasattr(args, 'bf') or args.bf == None: # Use default simple example
        ## You can create different bound for each dimension like this
        # lb = [-15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -15, -10, -3, -100, -40, -50]
        # ub = [15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 15, 10, 3, 20, 200, 1000]
        # problem_size = 18
        ## if you choose this way, the problem_size need to be same length as lb and ub

        ## Or bound is the same for all dimension like this
        lb = [-100]
        ub = [100]
        problem_size = 100
        ## if you choose this way, the problem_size can be anything you want

        ## Setting parameters
        obj_func = func_ackley
        verbose = True
        epoch = 1000
        pop_size = 50
    else:
        if not hasattr(args, 'dim'):
            dim = 10
        else:
            dim = args.dim
        bf_name_list = ['Hypersphere', 'Hyperellipsoid', 'Schwefel', 'Ackley', 'Michalewicz',
                        'Rastrigin', 'Rosenbrock', 'DeJong3', 'DeJong5', 'MartinGaddy',
                        'Griewank', 'Easom', 'GoldsteinAndPrice', 'PichenyGoldsteinAndPrice',
                        'StyblinskiTang', 'McCormick', 'Rana', 'EggHolder', 'Keane',
                        'Schaffer2', 'Himmelblau', 'PitsAndHoles']
        if args.bf in bf_name_list:
            bf_class = getattr(bf, args.bf)
            obj_func = bf_class(dim)
            if obj_func.minimum() == None:
                print(f"The dimension {dim} is not supported by {args.bf}, use default dimension {obj_func.n_dimensions()}")
                obj_func = bf_class()
                dim = obj_func.n_dimensions()
            if obj_func.minimum() == None:
                raise ValueError(f"Unable to create benchmark function {args.bf}")
            lb, ub = obj_func.suggested_bounds()
            problem_size = dim
        else:
            raise ValueError(f"Invalid benchmark function name {args.bf}, \
                             valid benchmark func names are {bf_name_list}")  
        verbose = True
        epoch = 1000
        pop_size = 50
    
    # Known minimum:
    print(f"Known minimum value: {obj_func.minimum().score}")

    md1 = BaseSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
    best_pos1, best_fit1, list_loss1 = md1.train()
    # return : the global best solution, the fitness of global best solution and the loss of training process in each epoch/iteration
    if md1.solution is not None:
        print(md1.solution[0])
        print(md1.solution[1])
    print(md1.loss_train)

    md2 = OriginalSMA(obj_func, lb, ub, problem_size, verbose, epoch, pop_size)
    best_pos2, best_fit2, list_loss2 = md2.train()
    # return : the global best solution, the fitness of global best solution and the loss of training process in each epoch/iteration
    print(best_pos2)
    print(best_fit2)
    print(list_loss2)