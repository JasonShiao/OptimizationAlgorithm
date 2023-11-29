import numpy as np
import re
from metaheuristic.ant_system import AntSystem, AntSystemMode, AntSystemOptions
import argparse

def generate_distance_matrix(size):
    random_matrix = np.random.rand(size, size)
    # Make the matrix symmetric
    distance_matrix = (random_matrix + random_matrix.T) / 2
    distance_matrix *= 100
    return distance_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data based on mode.')
    
    # Define the --mode argument with choices
    parser.add_argument('--mode', choices=[mode.value for mode in AntSystemMode],
                        help='Specify the processing mode (density, quantity, cycle)', required=True)
    args = parser.parse_args()

    mode_str = args.mode
    mode = AntSystemMode(mode_str)
    if mode == AntSystemMode.AntDensity:
        print('Processing in density mode...')
    elif mode == AntSystemMode.AntQuantity:
        print('Processing in quantity mode...')
    elif mode == AntSystemMode.AntCycle:
        print('Processing in cycle mode...')
    else:
        print('Invalid mode selected.')

    #f = open("data/tsp_problem_set/att48_d.txt")
    f = open("data/tsp_problem_set/gr17_d.txt")
    matrix_str = re.split(r'\s+', f.read())
    matrix_element_list = [float(item) for item in matrix_str if item != '']
    matrix_size = int(np.sqrt(len(matrix_element_list)))
    distance_matrix = np.array(matrix_element_list).reshape((matrix_size, matrix_size))
    
    n_ant = 300
    options = AntSystemOptions(AntSystemMode.AntCycle, n_ant, 100, 0.05, 1, 1)
    as_algo = AntSystem(options)
    as_algo.optimize(distance_matrix)

