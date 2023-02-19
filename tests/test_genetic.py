import unittest
from metaheuristic.genetic import bitwise_not, Encoder, GrayCodeEncoder, Chromosome, GA
import benchmark_functions as bf

def gen_grid_points(min, max, n_cut):
    """_summary_
    ref: https://stackoverflow.com/questions/45583274/how-to-generate-an-n-dimensional-grid-in-python

    Args:
        min (_type_): min bounds
        max (_type_): max bounds
        n_cut (_type_): number of cuts per dimension
    """
    import numpy as np
    s = tuple(slice(min[i], max[i], complex(0, n_cut)) for i in range(len(min)))
    return np.mgrid[s].reshape(len(min), -1).T.tolist()

class TestBitwiseNot(unittest.TestCase):
    def test_bitwise_not(self):
        self.assertEqual(bitwise_not(0, 8), 255)
        self.assertEqual(bitwise_not(30, 8), 225)
        self.assertEqual(bitwise_not(3, 5), 28)
        self.assertEqual(bitwise_not(0b0010100, 7), 0b1101011)

class TestGrayCodeEncoder(unittest.TestCase):
    def test_encode(self):
        self.assertEqual(GrayCodeEncoder([0, 0], [1, 1], [8, 8], 2).encode([0, 0]), [0, 0])
        self.assertEqual(GrayCodeEncoder([0, 0], [101, 32], [8, 8], 2).encode([79, 10]), [164, 120])
    def test_decode(self):
        val = GrayCodeEncoder([0, 0], [101, 32], [8, 8], 2).decode([164, 120])
        self.assertAlmostEqual(val[0], 78.8, delta=0.1)
        self.assertAlmostEqual(val[1], 10.0, delta=0.1)

class TestChromosome(unittest.TestCase):
    encoder = GrayCodeEncoder([0, 0], [101, 32], [8, 8], 2)
    def test_init(self):
        chromosome = Chromosome([79, 10], self.encoder)
        self.assertEqual(chromosome.encoded_genes, [164, 120])
        self.assertAlmostEqual(chromosome.genes[0], 78.8, delta=0.1)
        self.assertAlmostEqual(chromosome.genes[1], 10.0, delta=0.1)
    def test_set_genes(self):
        chromosome = Chromosome([0, 0], self.encoder)
        chromosome.genes = [33.9, 22]
        self.assertEqual(chromosome.encoded_genes, [125, 248])
        self.assertAlmostEqual(chromosome.genes[0], 34.0, delta=0.1)
        self.assertAlmostEqual(chromosome.genes[1], 21.9, delta=0.1)
    def test_set_encoded_genes(self):
        chromosome = Chromosome([0, 0], self.encoder)
        chromosome.encoded_genes = [125, 248]
        self.assertEqual(chromosome.encoded_genes, [125, 248])
        self.assertAlmostEqual(chromosome.genes[0], 34.0, delta=0.1)
        self.assertAlmostEqual(chromosome.genes[1], 21.9, delta=0.1)


class TestGeneticAlgorithmMethods(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_dim = 2
        self.func1 = bf.Schwefel(n_dimensions=self.n_dim)
        self.min_bounds, max_bounds = self.func1.suggested_bounds()
        self.init_points = gen_grid_points(self.min_bounds, max_bounds, 5)
        self.encoder = GrayCodeEncoder(self.min_bounds, max_bounds, [8] * len(self.min_bounds), self.n_dim)
    
    def test_crossover(self):
        pass
    
    def test_upadte_roulette(self):
        pass
    
    def test_optimize(self):
        print('test_optimize')
        init_populations = [Chromosome(point, self.encoder) for point in self.init_points]
        ga_optimizer = GA()
        ga_optimizer.optimize(self.func1, init_populations, n_dim=self.n_dim)
        pass

if __name__ == '__main__':
    unittest.main()