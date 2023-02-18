import graycode
import numpy as np
from random import random

# 1. Encode input
# 2. 
# graycode.tc_to_gray_code(i)
# graycode.gray_code_to_tc(i)

class GA:
    input_min = None
    input_max = None
    num_bit_gene = 0 # number of bits per gene (encoded)
    n_dim = 0
    graycode_genes = None # genes represented with graycode value
    fitnesses = None
    find_min = False
    def __init__(self):
        pass
    def roulette_select(self):
        # self.graycode_genes
        if self.find_min:
            invert_fitnesses = [1 / fitness for fitness in self.fitnesses]
            roulette_map = []
            accumulate = 0
            for invert_fitness in invert_fitnesses:
                accumulate += invert_fitness
                roulette_map.append(accumulate)
            roulette_map = [val / accumulate for val in roulette_map]
        else: # find max
            roulette_map = []
            accumulate = 0
            for fitness in self.fitnesses:
                accumulate += fitness
                roulette_map.append(accumulate)
            roulette_map = [val / accumulate for val in roulette_map]
            pointer = random()
            
        pass
    def mutate(self, gene):
        pass
    def crossover(self, gene_1, gene_2, mask):
        """Based on truth table, the gene_1' = (mask & gene_2) | (gene_1 & bitwise_not(mask))
                                     gene_2' = (gene_1 & mask) | (bitwise_not(mask) & gene_2)
        Args:
            gene_1 (_type_): _description_
            gene_2 (_type_): _description_
        """
        new_gene_1 = [(mask[d] & gene_2[d]) | (gene_1[d] & self.bitwise_not(mask)[d]) for d in range(self.n_dim)]
        new_gene_2 = [(gene_1[d] & mask[d]) | (self.bitwise_not(mask)[d] & gene_2[d]) for d in range(self.n_dim)]
        return new_gene_1, new_gene_2

    def bitwise_not(self, gene):
        # NOTE: Cannot directly get from ~x which is two's complement result (and will have negative value)
        #       the result for unsigned 'not' operation depends on the number of bits considered
        #       ref: https://stackoverflow.com/a/64405631
        return [2 ** self.num_bit_gene[d] - gene[d] - 1 
                for d in range(self.n_dim)]
    
    def encode(self, x):
        """ Encode original real value into Graycode value
        Args:
            x (_type_): original real value
        Returns:
            _type_: graycode encoded value
        """
        # 1. Cip with min and max
        x = np.array(x).clip(self.input_min, self.input_max).tolist()
        print(x)
        # 2. Encode float into int: dec / (dec_max - dec_min) = float / (f_max - f_min) -> (2 ** self.num_bit_gene - 1) * (x - min) / (max - min)
        x_dec = [round((2 ** self.num_bit_gene[d] - 1) * (x[d] - self.input_min[d]) / (self.input_max[d] - self.input_min[d])) 
                 for d in range(self.n_dim)]
        # 3. encode with Graycode
        x_encoded = [graycode.tc_to_gray_code(x_dec[d]) for d in range(self.n_dim)]
        return x_encoded

    def decode(self, x):
        """ Decode Graycode value into original real value
        Args:
            x (_type_): graycode encoded value
        Returns:
            _type_: real value
        """
        x_dec = [graycode.gray_code_to_tc(x[d]) for d in range(self.n_dim)]
        x_val = [(self.input_max[d] - self.input_min[d]) * x_dec[d] / (2 ** self.num_bit_gene[d] - 1) + self.input_min[d]
                    for d in range(self.n_dim)]
        return x_val
    pass
