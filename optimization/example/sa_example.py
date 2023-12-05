from optimization.metaheuristic.simulated_annealing import SA, CoolingSchedule, NextStepPolicy
import benchmark_functions as bf

def sa_example(args):
    # Determine benchmark function, dimension and bounds
    if args == None or not hasattr(args, 'bf') or args.bf == None: # Use default simple example
        ## You can create different bound for each dimension like this
        raise ValueError("Please specify benchmark function name")
    
    bf_name_list = ['Hypersphere', 'Hyperellipsoid', 'Schwefel', 'Ackley', 'Michalewicz',
                        'Rastrigin', 'Rosenbrock', 'DeJong3', 'DeJong5', 'MartinGaddy',
                        'Griewank', 'Easom', 'GoldsteinAndPrice', 'PichenyGoldsteinAndPrice',
                        'StyblinskiTang', 'McCormick', 'Rana', 'EggHolder', 'Keane',
                        'Schaffer2', 'Himmelblau', 'PitsAndHoles']
    if not hasattr(args, 'dim'):
        dim = 10
    else:
        dim = args.dim
    if args.bf in bf_name_list:
        bf_class = getattr(bf, args.bf)
        obj_func = bf_class(args.dim)
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
    max_iter_per_temp = 50
    sa = SA(100, 0.1, CoolingSchedule.Scale, 0.92, epoch, max_iter_per_temp, NextStepPolicy.Cauchy)
    # Set tolerance for different benchmark functions with different minimum score
    #if obj_func.minimum().score < -0.9:
    #    acceptable_score = obj_func.minimum().score * 0.98
    #elif obj_func.minimum().score < 0:
    #    acceptable_score = obj_func.minimum().score * 0.9
    #elif obj_func.minimum().score < 1:
    #    acceptable_score = obj_func.minimum().score * 1.1
    #else:
    #    acceptable_score = obj_func.minimum().score * 1.02
    sa.optimize(obj_func, problem_size, verbose)
