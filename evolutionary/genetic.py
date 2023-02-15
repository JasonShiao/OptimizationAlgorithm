import graycode

# 1. Encode input
# 2. 
# graycode.tc_to_gray_code(i)
# graycode.gray_code_to_tc(i)

class GA:
    input_min = None
    input_max = None
    num_bit_gene = 0 # number of bits per gene (encoded)
    n_dim = 0
    def __init__(self):
        pass
    def mutate(gene):
        pass
    def crossover(gene_1, gene_2):
        pass
    def encode(self, x):
        # 1. Cip with min and max
        # 2. Convert from float to int: (2 ** self.num_bit_gene - 1) * (x - min) / (max - min)
        # x_dec = [ (2 ** self.num_bit_gene - 1) * (x[d] - self.input_min[d]) / (self.input_max[d] - self.input_min[d]) 
        #         for d in range(self.n_dim)]
        # 3. encode with Graycode
        # x_encoded = [graycode.tc_to_gray_code(x_dec[d]) for d in range(self.n_dim)]
        #return x_encoded
        pass
    def decode(self, x):
        """_summary_

        Args:
            x (_type_): graycode encoded value

        Returns:
            _type_: _description_
        """
        x_dec = [graycode.gray_code_to_tc(x[d]) for d in range(self.n_dim)]
        x_val = [(self.input_max[d] - self.input_min[d]) * x_dec[d] / (2 ** self.num_bit_gene) + self.input_min[d]
                    for d in range(self.n_dim)]
        return x_val
    pass
