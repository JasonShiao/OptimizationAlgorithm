import argparse
from optimization.example.aco_example import tsp_example
from optimization.metaheuristic.ant_system import AntSystemVariant    
    

def main():
    parser = argparse.ArgumentParser(description='Process data based on mode.')
    
    # Define the --mode argument with choices
    parser.add_argument('--mode', choices=[mode.value for mode in AntSystemVariant],
                        help='Specify the processing mode (density, quantity, cycle)', required=True)
    parser.add_argument('--problem', choices=['att48', 'gr17', 'fri26', 'dantzig42', 'p01'],
                        help='Specify the problem set (att48, gr17, fri26, dantzig42, p01)', required=True)
    args = parser.parse_args()
    tsp_example(args)
