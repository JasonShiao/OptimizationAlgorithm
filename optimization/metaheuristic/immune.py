import graycode
import numpy as np
import random
import math

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
        # 2. Encode float into int
        ranges = [self.max_bounds[d] - self.min_bounds[d] for d in range(self.n_dim)]
        x_integer_encoded = [round((2 ** self.n_bits[d] - 1) * (x_clipped[d] - self.min_bounds[d]) / ranges[d]) 
                 for d in range(self.n_dim)]
        # 3. encode with Graycode
        x_graycode_encoded = [graycode.tc_to_gray_code(x_integer_encoded[d]) for d in range(self.n_dim)]
        return x_graycode_encoded
    
    def decode(self, x):
        ranges = [self.max_bounds[d] - self.min_bounds[d] for d in range(self.n_dim)]
        x_integer_encoded = [graycode.gray_code_to_tc(x[d]) for d in range(self.n_dim)]
        x_recovered = [ranges[d] * x_integer_encoded[d] / (2 ** self.n_bits[d] - 1) + self.min_bounds[d]
                    for d in range(self.n_dim)]
        return x_recovered


class BCell:
    encoder = None
    def __init__(self, antigens, encoder, encoded_antigens=False):
        self.encoder = encoder
        if encoded_antigens:
            self.encoded_antigens = antigens
        else:
            self.antigens = antigens
    
    @property
    def antigens(self):
        """ Real value angigens of B-Cell in each dimension
        Returns:
            _type_: _description_
        """
        return self._antigens
    @antigens.setter
    def antigens(self, val):
        """Not only encode the values but also apply constraints (min, max) on the value
        Args:
            val (_type_): _description_
        """
        self.encoded_antigens = self.encoder.encode(val)
        self._antigens = self.encoder.decode(self.encoded_antigens)
    
    def __str__(self):
        return str(self.antigens)
    
    @property
    def encoded_antigens(self):
        return self._encoded_antigens
    @encoded_antigens.setter
    def encoded_antigens(self, val):
        self._encoded_antigens = val
        self._antigens = self.encoder.decode(self.encoded_antigens)


class IA:
    """Immune Algorithm
    """
    n_dim = 0
    find_min = False
    population = [] # population of B-Cells represented with graycode encoded antignes
    fitnesses = []
    roulette_map = []
    encoder = None
    def __init__(self):
        pass
    def upadte_roulette(self):
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
    
    def mutate(self, chr, mutate_rate=0.05):
        encoded_antigens = chr.encoded_antigens
        for d in range(self.n_dim):
            for i in range(self.encoder.n_bits[d]):
                if random.random() < mutate_rate:
                    encoded_antigens[d] = encoded_antigens[d] ^ (1 << i)
        chr.encoded_antigens = encoded_antigens
        return chr
    
    def crossover(self, b_cell_1, b_cell_2, masks):
        """Based on truth table, the b_cell_1' = (mask & b_cell_2) | (b_cell_1 & bitwise_not(mask))
                                     b_cell_2' = (b_cell_1 & mask) | (bitwise_not(mask) & b_cell_2)
        Args:
            b_cell_1 (_type_): first B-cell
            b_cell_2 (_type_): second B-cell
            mask (_type_): mask for crossover
        Returns: two child B-Cells
        """
        new_antigens_for_b_cell_1 = [(masks[d] & b_cell_2.encoded_antigens[d]) | (b_cell_1.encoded_antigens[d] & bitwise_not(masks[d], b_cell_1.encoder.n_bits[d])) 
                               for d in range(self.n_dim)]
        new_b_cell_first = BCell(new_antigens_for_b_cell_1, self.encoder, True)
        new_antigens_for_b_cell_2 = [(b_cell_1.encoded_antigens[d] & masks[d]) | (bitwise_not(masks[d] & b_cell_2.encoded_antigens[d], b_cell_2.encoder.n_bits[d])) 
                               for d in range(self.n_dim)]
        new_b_cell_second = BCell(new_antigens_for_b_cell_2, self.encoder, True)
        
        return new_b_cell_first, new_b_cell_second
    
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
    
    def optimize(self, objective_func, initial_population, n_dim, find_min = True, mutate_rate = 0.05, elitism = 0.1):
        # 0. Initialize
        if len(initial_population) == 0:
            return # TODO: Randomly generate initial BCells
        self.fitnesses = []
        self.roulette_map = []
        self.n_dim = n_dim
        self.population = initial_population
        self.encoder = self.population[0].encoder
        self.find_min = find_min # Typical optimization problem in benchmark_function is to find min
        # 1. evaluate fitness value
        self.fitnesses = [objective_func(self.population[idx].antigens) for idx in range(len(self.population))]
        # 2. update roulette
        self.upadte_roulette()
        masks = self.gen_two_point_crossover_masks(self.n_dim, self.encoder.n_bits)
        not_improve_cnt = 0
        round_cnt = 0
        best_fitness = math.inf
        while True:  
            elite_list = []
            # 1. Elitism selection
            sort_index = np.argsort(np.array(self.fitnesses))
            num_elite = round(elitism * len(self.population))
            if len(self.population) - num_elite % 2 == 1:
                num_elite += 1
            for i in range(num_elite):
                elite_list.append(self.population[sort_index[i]])
            # 2. roulette select parents for next round
            selected_parents_idx = [self.roulette_select(2) 
                                     for i in range(math.floor((len(self.population) - num_elite) / 2))]
            # 3. crossover
            children_list = []
            for i in range(len(selected_parents_idx)):
                children_list += self.crossover(self.population[selected_parents_idx[i][0]], 
                                       self.population[selected_parents_idx[i][1]],
                                       masks)
            # 4. mutate
            children_list = [self.mutate(chr, mutate_rate) for chr in children_list]
            # 5 Update population
            self.population = elite_list + children_list
            # 6. evaluate fitness value
            self.fitnesses = [objective_func(self.population[idx].antigens) for idx in range(len(self.population))]
            #print(min(self.fitnesses))
            # 7. update roulette
            self.upadte_roulette()
            # 8. Check termination condition
            if best_fitness > min(self.fitnesses):
                best_fitness = min(self.fitnesses)
                not_improve_cnt = 0
            else:
                not_improve_cnt += 1
            round_cnt += 1
            if not_improve_cnt > 100:
                break
