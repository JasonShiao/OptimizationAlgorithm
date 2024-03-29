import numpy as np
import re
import cProfile
from optimization.metaheuristic.ant_system import AntSystem, AntSystemOptions, AntSystemVariant
from optimization.metaheuristic.ant_colony_system import AntColonySystem, AntColonySystemOptions
from optimization.misc.utils import tsplib95_get, get_weight_matrix_from_tsplib, TSPLIB95Category

cprofiler = cProfile.Profile()

"""
  In AS a good heuristic to initialize the pheromone
  trails is to set them to a value slightly higher than the expected amount of pheromone
  deposited by the ants in one iteration; a rough estimate of this value can be obtained
  by setting, Eði; jÞ, tij ¼ t0 ¼ m=Cnn, where m is the number of ants, and Cnn is the
  length of a tour generated by the nearest-neighbor heuristic (in fact, any other reasonable tour construction procedure would work fine).
"""

"""
Concerning solution construction, there are two di¤erent ways of implementing it: parallel and sequential solution construction. 
In the parallel implementation, at each construction step all the ants move from their current city to the next one, 
while in the sequential implementation an ant builds a complete tour before the next one starts to build another one. 
In the AS case, both choices for the implementation of the tour construction are equivalent in the sense that 
they do not significantly influence the algorithm’s behavior. 
As we will see, this is not the case for other ACO algorithms such as ACS.
"""

"""
As we said, the relative performance of AS when compared to other metaheuristics tends to decrease dramatically as the size of 
the test-instance increases. Therefore, a substantial amount of research on ACO has focused on how to improve AS.
"""

def generate_distance_matrix(size):
    random_matrix = np.random.rand(size, size)
    # Make the matrix symmetric
    distance_matrix = (random_matrix + random_matrix.T) / 2
    distance_matrix *= 100
    return distance_matrix


def tsp_example(args):
    mode_str = args.mode
    mode = AntSystemVariant(mode_str)
    if mode == AntSystemVariant.AntDensity:
        print('Processing in density mode...')
    elif mode == AntSystemVariant.AntQuantity:
        print('Processing in quantity mode...')
    elif mode == AntSystemVariant.AntCycle:
        print('Processing in cycle mode...')
    else:
        print('Invalid mode selected.')
    
    problem, optimal_sol = tsplib95_get(TSPLIB95Category.TSP, args.problem)
    try:
        distance_matrix = get_weight_matrix_from_tsplib(problem)
    except Exception as e:
        # TODO: Set as None?
        distance_matrix = np.ones((10, 10))
    if problem == None:
        print(f"Get Problem {args.problem} failed")
        raise FileNotFoundError
    
    print(f"Distance matrix = {distance_matrix}")
    #n_ant = 300
    #options = AntSystemOptions(AntSystemMode.AntCycle, n_ant, 100, 0.05, 1, 1)
    #as_algo = AntSystem(options)
    #as_algo.optimize(distance_matrix)
    
    # Ant System
    n_dim = distance_matrix.shape[0]
    n_ant = n_dim
    rho = 0.5
    alpha = 1
    beta = 2
    # TODO: Handle np.inf case (0 exists in the matrix)
    #tau_0 = n_ant / (np.min(distance_matrix))
    tau_0 = n_ant / np.mean(distance_matrix)
    options = AntSystemOptions(mode, n_ant, 500, tau_0, rho, alpha, beta, 1)
    #options = AntSystemOptions(AntSystemMode.AntColonySystem, n_ant, 100, 0.05, 1, 1, 1, 0, np.inf, 0.03, 1, 0.2)
    #options = AntSystemOptions(AntSystemMode.MinMaxAntSystem, n_ant, 100, 0.05, 1, 1, 1, 0, 100, 0.03, 1, 0.2)
    #options = AntSystemOptions(AntSystemVariant.AntDensity, n_ant, 100, tau_0, 0.5, 1, 2, 1)
    #options = AntSystemOptions(AntSystemVariant.AntQuantity, n_ant, 100, tau_0, 0.2, 1, 2, 1)
    #options = AntSystemOptions(AntSystemVariant.AntCycle, n_ant, 100, tau_0, 0.2, 1, 2, 1)
    #options = AntSystemOptions(AntSystemMode.AntQuantity, n_ant, 100, 0.05, 1, 2)
    #options = AntSystemOptions(AntSystemMode.AntDensity, n_ant, 100, 0.05, 1, 2)
    as_algo = AntSystem()
    if args.profile:
        cprofiler.enable()
    as_algo.optimize(n_dim, distance_matrix, options, problem.get_graph() if 'node_coords' in problem.as_name_dict() else None)
    if args.profile:
        cprofiler.disable()
        cprofiler.print_stats(sort='cumulative')
    # Ant Colony System
    #phi = 0.1 # by experiment, a good phi = 0.1
    #tau_0 = 1 / (n_ant * np.min(distance_matrix))
    #rho = 0.1 # good value for ACS; for AS, rho = 0.5
    #aco = AntColonySystem(distance_matrix)
    #options = AntColonySystemOptions(AntSystemVariant.AntColonySystem, n_ant, 100, tau_0, rho, 1, 2, 1, 0.2, phi)
    #cprofiler.enable()
    #aco.optimize(options)
    #cprofiler.disable()
    #cprofiler.print_stats(sort='cumulative')

