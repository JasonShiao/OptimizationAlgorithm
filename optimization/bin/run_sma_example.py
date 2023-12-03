from optimization.example.SMA_example import sma_example
import argparse

benchmark_function_list = ['Hypersphere', 'Hyperellipsoid', 'Schwefel', 'Ackley', 'Michalewicz',
                        'Rastrigin', 'Rosenbrock', 'DeJong3', 'DeJong5', 'MartinGaddy',
                        'Griewank', 'Easom', 'GoldsteinAndPrice', 'PichenyGoldsteinAndPrice',
                        'StyblinskiTang', 'McCormick', 'Rana', 'EggHolder', 'Keane',
                        'Schaffer2', 'Himmelblau', 'PitsAndHoles']

def main():
    # input arguments: bf, dim
    parser = argparse.ArgumentParser(description='Test benchmark function with SMA optimization.')
    parser.add_argument('--bf', choices = benchmark_function_list,
                        help = 'Specify the benchmark function to be tested (default: Ackley)', required=False)
    parser.add_argument('--dim', type = int, default = 10,
                        help = 'Specify the dimension of the benchmark function (default: 10)', required=False)
    args = parser.parse_args()
    sma_example(args)
