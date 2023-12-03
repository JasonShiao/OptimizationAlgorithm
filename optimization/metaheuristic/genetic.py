import graycode
import numpy as np
import random
import math
from typing import List, Dict, Tuple
import copy
    
def bitwise_not(x, n_bits):
    """bitwise not the encoded value
    Returns:
        _type_: _description_
    """
    # NOTE: Cannot directly get from ~x which is two's complement result (and will have negative value)
    #       the result for unsigned 'not' operation depends on the number of bits considered
    #       ref: https://stackoverflow.com/a/64405631
    return 2 ** n_bits - x - 1


class Encoder:
    n_dim = 0
    min_bounds = []
    max_bounds = []
    n_bits = [] # number of bits in each dimension
    def __init__(self, min_bounds, max_bounds, n_bits, n_dim):
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.n_bits =  n_bits
        self.n_dim = n_dim
        pass
    def encode(self, x):
        pass
    def decode(self, x):
        pass

class GrayCodeEncoder(Encoder):
    def __init__(self, min_bounds, max_bounds, n_bits, n_dim):
        super().__init__(min_bounds, max_bounds, n_bits, n_dim)
    def encode(self, x):
        # 1. Cip with min and max
        x_clipped = np.array(x).clip(self.min_bounds, self.max_bounds).tolist()
        # 2. Encode float into int: dec / (dec_max - dec_min) = float / (f_max - f_min) -> (2 ** self.num_genes - 1) * (x - min) / (max - min)
        ranges = [self.max_bounds[d] - self.min_bounds[d] for d in range(self.n_dim)]
        x_integer_encoded = [round((2 ** self.n_bits[d] - 1) * (x_clipped[d] - self.min_bounds[d]) / ranges[d]) 
                 for d in range(self.n_dim)]
        # 3. encode with Graycode
        x_graycode_encoded = [graycode.tc_to_gray_code(x_integer_encoded[d]) for d in range(self.n_dim)]
        return x_graycode_encoded
    
    def decode(self, x):
        """
            NOTE: The decoded value is mapped by graycode table, not the ordinary binary table
        """
        ranges = [self.max_bounds[d] - self.min_bounds[d] for d in range(self.n_dim)]
        x_integer_encoded = [graycode.gray_code_to_tc(x[d]) for d in range(self.n_dim)]
        x_recovered = [ranges[d] * x_integer_encoded[d] / (2 ** self.n_bits[d] - 1) + self.min_bounds[d]
                    for d in range(self.n_dim)]
        return x_recovered


class Chromosome:
    def __init__(self, genes, encoder: Encoder, encoded_genes=False):
        self.encoder = encoder
        if encoded_genes:
            self.encoded_genes = genes
        else:
            self.genes = genes
    
    @property
    def genes(self):
        """ Real value genes of chromosome in each dimension
        Returns:
            _type_: _description_
        """
        return self._genes
    @genes.setter
    def genes(self, val):
        """Not only encode the values but also apply constraints (min, max) on the value
        Args:
            val (_type_): _description_
        """
        if self.encoder is not None:
            self.encoded_genes = self.encoder.encode(val)
            self._genes = self.encoder.decode(self.encoded_genes)
    
    def __str__(self):
        return str(self.genes)
    
    @property
    def encoded_genes(self):
        return self._encoded_genes
    @encoded_genes.setter
    def encoded_genes(self, val):
        self._encoded_genes = val
        if self.encoder is not None:
            self._genes = self.encoder.decode(self.encoded_genes)
    

class GA:
    """Genetic Algorithm with Elitism and Roulette Selection

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    n_dim = 0
    find_min = False
    population = [] # population of chromosomes represented with graycode encoded genes
    fitnesses = []
    roulette_map = []
    #encoder = None
    def __init__(self):
        pass
    def upadte_roulette(self):
        """ 
            The probability of selecting a chromosome is proportional to its fitness value
        """
        self.roulette_map = []
        if self.find_min:
            invert_fitnesses = [1 / fitness for fitness in self.fitnesses]
            accumulate = 0
            for invert_fitness in invert_fitnesses:
                accumulate += invert_fitness
                self.roulette_map.append(accumulate)
            self.roulette_map = [val / accumulate for val in self.roulette_map]
        else: # find max
            accumulate = 0
            for fitness in self.fitnesses:
                accumulate += fitness
                self.roulette_map.append(accumulate)
            self.roulette_map = [val / accumulate for val in self.roulette_map]

    def roulette_select(self, n):
        next_parents = []
        for i in range(n):
            pointer = random.random()
            for idx in range(len(self.roulette_map)):
                if self.roulette_map[idx] > pointer:
                    next_parents.append(idx)
                    break
        if len(next_parents) != n:
            raise Exception("Amount of selected parents should match n")
        return next_parents
    
    def mutate(self, chr: Chromosome, mutate_rate: float = 0.05):
        """ Mutate the chromosome in place

        Args:
            chr (Chromosome): _description_
            mutate_rate (float, optional): _description_. Defaults to 0.05.
        """
        encoded_genes = chr.encoded_genes
        for d in range(chr.encoder.n_dim):
            for i in range(chr.encoder.n_bits[d]):
                if random.random() < mutate_rate:
                    encoded_genes[d] = encoded_genes[d] ^ (1 << i)
                    #print(f" mutate triggered: {encoded_genes[d]}")
        # Must set to update the genes property
        chr.encoded_genes = encoded_genes
    
    def crossover(self, chr_1, chr_2, masks):
        # TODO: crossover with probability (0.0 - 1.0)
        """Based on truth table, the chr_1' = (mask & chr_2) | (chr_1 & bitwise_not(mask))
                                     chr_2' = (chr_1 & mask) | (bitwise_not(mask) & chr_2)
        Args:
            chr_1 (_type_): first chromosome
            chr_2 (_type_): second chromosome
            mask (_type_): mask for crossover
        Returns: two child chromosomes
        """
        new_genes_for_chr_1 = [(masks[d] & chr_2.encoded_genes[d]) | (chr_1.encoded_genes[d] & bitwise_not(masks[d], chr_1.encoder.n_bits[d])) 
                               for d in range(chr_1.encoder.n_dim)]
        new_chr_1 = Chromosome(new_genes_for_chr_1, chr_1.encoder, True)
        new_genes_for_chr_2 = [(chr_1.encoded_genes[d] & masks[d]) | (bitwise_not(masks[d] & chr_2.encoded_genes[d], chr_2.encoder.n_bits[d])) 
                               for d in range(chr_2.encoder.n_dim)]
        new_chr_2 = Chromosome(new_genes_for_chr_2, chr_2.encoder, True)
        
        return new_chr_1, new_chr_2
    
    def gen_two_point_crossover_masks(self, n_dim, n_bits):
        """Generate two point crossover masks for each dimension

        Args:
            n_dim (_type_): number of dimensions
            n_bits (_type_): number of bits for each dimension

        Returns:
            _type_: list of masks
        """
        masks = []
        for d in range(n_dim):
            low, high = sorted(random.sample(range(n_bits[d]), 2))
            mask = 0b0
            for i in range(high - low):
                mask += 0b1 << i
            masks.append(mask << low)
            #masks.append([0] * low + [1] * (high - low) + [0] * (n_bits[d] - high))
        return masks
    
    def optimize(self, objective_func, initial_population: List[Chromosome], find_min: bool = True, mutate_rate: float = 0.05, eliticism: float = 0.1):
        # Check (Validate) input
        if not isinstance(initial_population, list) or not all(isinstance(item, Chromosome) for item in initial_population):
            raise Exception("Initial population should be a list of Chromosome")
        if len(initial_population) == 0:
            raise Exception("Initial population should not be empty")
        # 0. Initialize
        self.fitnesses = []
        self.roulette_map = []
        #self.n_dim = initial_population[0].encoder.n_dim
        self.population = initial_population
        self.find_min = find_min # Typical optimization problem in benchmark_function is to find min
        masks = self.gen_two_point_crossover_masks(initial_population[0].encoder.n_dim, self.population[0].encoder.n_bits)
        not_improve_cnt = 0
        round_cnt = 0
        best_fitness = math.inf if self.find_min else -math.inf
        while True:  
            # 1. evaluate fitness value with current population
            self.fitnesses = [objective_func(self.population[idx].genes) for idx in range(len(self.population))]
            # 2. update roulette
            self.upadte_roulette()
            # 3. Update best and Check termination condition
            if not self.find_min and best_fitness > max(self.fitnesses):
                best_fitness = max(self.fitnesses)
                not_improve_cnt = 0
            elif self.find_min and best_fitness > min(self.fitnesses):
                best_fitness = min(self.fitnesses)
                not_improve_cnt = 0
            else:
                not_improve_cnt += 1
            if not_improve_cnt > 100:
                break
            
            elite_list = []
            # 4. Eliticism selection
            sort_index = np.argsort(np.array(self.fitnesses))
            num_elite = round(eliticism * len(self.population))
            if len(self.population) - num_elite % 2 == 1:
                num_elite += 1
            for i in range(num_elite):
                elite_list.append(self.population[sort_index[i]])
            # 5. roulette select parents for this round of evolution
            selected_parents_idx = [self.roulette_select(2) 
                                     for i in range(math.floor((len(self.population) - num_elite) / 2))]
            # 6. crossover
            children_list = []
            # TODO: crossover with probability
            for i in range(len(selected_parents_idx)):
                children_list += self.crossover(self.population[selected_parents_idx[i][0]], 
                                       self.population[selected_parents_idx[i][1]],
                                       masks)
            # 7. mutate
            for chr in children_list:
                self.mutate(chr, mutate_rate)
            # 8 Update population (Replace the entire population by the child chromosomes as the parent generation)
            self.population = elite_list + children_list

            # Record evolution round count
            round_cnt += 1
            
        #print(round_cnt)
