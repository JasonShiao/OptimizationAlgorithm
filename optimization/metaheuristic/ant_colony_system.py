# Refined algorithm based on the Ant System algorithm
# Dorigo et al. in 1996, Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem
import numpy as np
from optimization.metaheuristic.ant_system import AntSystemVariant, AntSystemOptions, AntSystemBase

class AntColonySystemOptions(AntSystemOptions):
    def __init__(self, mode: AntSystemVariant, n_ant: int, n_iter: int, tau_0: float, 
                rho: float, alpha: float, beta: float, Q: float,
                q_0: float, phi: float):
        """_summary_

        Args:
            mode (AntSystemVariant): _description_
            n_ant (int): typically = n_pos
            n_iter (int): _description_
            tau_0 (float): by experiment, a good tau_0 = 1 / (n_pos * C^nn)
            rho (float): 0 < rho < 1 for global update
            alpha (float): typically 1
            beta (float): typically 2-5
            q_0 (float): 0 < q_0 < 1
            phi (float): 0 < phi < 1 for local update, by experiment, a good phi = 0.01
        """
        super().__init__(mode, n_ant, n_iter, tau_0, rho, alpha, beta, Q)
        if q_0 < 0 or q_0 > 1:
            raise ValueError("q_0 must be in [0, 1]")
        if phi < 0 or phi > 1:
            raise ValueError("phi must be in [0, 1]")
        self.q_0 = q_0
        self.phi = phi

class ArtificialAnt:
    def __init__(self, n_pos, init_pos):
        self.current_pos = init_pos
        self.n_pos = n_pos
        self.path = [] # same as tabu list
        self.path.append(self.current_pos)
        self.move_distance = 0
    
    def complete(self):
        return len(self.path) == self.n_pos
    
    def move(self, next_pos, distance):
        self.path.append(next_pos)
        self.move_distance += distance
        self.current_pos = next_pos
    
    def release_memory(self):
        """Release memory of path but keep the init position
        """
        self.current_pos = self.path[0]
        self.path = []
        self.path.append(self.current_pos)
        self.move_distance = 0

        

class AntColonySystem(AntSystemBase):
    """
        In addition to the original Ant System, ACS has the following features:
        1. With local pheromone update:
              tau_ij = (1 - phi) * tau_ij + rho * tau_0
              where phi is the pheromone decay coefficient
            
            Implicit pheromone limit: (automatically restricted by the pheromone update rule)
             tau_0 < pheromone < 1 / C^bs
             
        2. The offline pheromone update is performed at the end of each iteration 
              and only the best ant can update pheromone (similar to MMAS)
              tau = (1 - rho) * tau + rho * delta(tau_best)
        3. Pseudorandom proportional rule: 
             the probability for an ant to move from city i to city j depends on 
             a random variable q uniformly distributed over [0, 1], and a para- meter q_0;if q <= q0
        4. Candidate list:
                the next city is chosen from a restricted candidate list,
                for city outside the list, it will only be chosen if all cities in the candidate list have been visited
        
    """
    def __init__(self, distance_matrix: np.ndarray):
        super().__init__(distance_matrix)
    
    def move(self, ant: ArtificialAnt, next_pos: int, distance_matrix: np.ndarray):
        pass
    
    #def transition_rule(self, ant: ArtificialAnt, pheromone: np.ndarray, heuristic_info: np.ndarray, candidate_list: np.ndarray):
    #    """Transition rule for ACS
    #    """
    #    # Check whether all candidates in candidate list have been visited (subset of ant.path)
    #    candidate_list_subset = [candidate for candidate in candidate_list if candidate not in ant.path]
    #    if len(candidate_list_subset) > 0:
    #        # Calculate the probability for each candidate
    #        prob = np.zeros(len(candidate_list_subset))
    #    else: # Choose nodes from outside the candidate list
    #        candidate_list_subset = [candidate for candidate in range(self.n_pos) if candidate not in ant.path]
    #    # Try Select from candidate list first
    
    def transition_rule(self, ant: ArtificialAnt, pheromone: np.ndarray):
        """Transition rule for ACS
        """
        # Calculate the probability for each candidate
        #print(f"pheromone = {pheromone}")
        #print(f"heuristic_info = {self.heuristic_info}")
        #print(f"alpha = {self.options.alpha}, beta = {self.options.beta}")
        p: np.ndarray = (pheromone[ant.current_pos] ** self.options.alpha) * (self.heuristic_info[ant.current_pos] ** self.options.beta)
        #print(f"p = {p}")
        # Apply tabu list
        p[ant.path] = 0
        # get random q and compare with q_0
        q = self.rng.random()
        if q <= self.options.q_0:
            # Choose the best candidate
            next_pos = np.argmax(p)
        else:
            # As normal AS
            normalized_p = p / np.sum(p)
            #print(f"normalized_p = {normalized_p}")
            next_pos = self.rng.choice(np.arange(self.n_pos), p = normalized_p)
        return next_pos
        
        
    def optimize(self, options: AntColonySystemOptions):
        if options.algo_variant != AntSystemVariant.AntColonySystem:
            raise ValueError("Invalid variant for Ant Colony System")
        self.options = options
        # Initialize ants
        pheromone = np.full(self.distance_matrix.shape, self.options.tau_0)
        ants: list[ArtificialAnt] = []
        nodes = [[] for i in range(self.n_pos)] # nodes[i][k] is the k-th ant index at node i (i.e. len(nodes[i]) = b_i in original paper)
        CANDIDATE_SPACE = 10 if self.n_pos > 10 else self.n_pos
        candidate_list = np.argsort(self.heuristic_info)[:, :CANDIDATE_SPACE] # candidate_list[i] is the candidate list for node i
        # Init pos (node) for each ant
        for i in range(self.options.n_ant):
            # randomly choose the start node
            start_node = np.random.choice(np.arange(self.n_pos))
            nodes[start_node].append(i)
            # Initialize ant
            ants.append(ArtificialAnt(self.n_pos, start_node))
        for i in range(self.options.max_iter):
            while True:
                next_step_nodes = [[] for i in range(self.n_pos)]
                for depart_node_idx in range(len(nodes)):
                    for ant_idx in nodes[depart_node_idx]:
                        #next_pos = self.transition_rule(ants[ant_idx], pheromone, self.heuristic_info, candidate_list[depart_node_idx])
                        next_pos = self.transition_rule(ants[ant_idx], pheromone)
                        ants[ant_idx].move(next_pos, self.distance_matrix[ants[ant_idx].current_pos][next_pos])
                        next_step_nodes[next_pos].append(ant_idx)
                        # Local update pheromone
                        pheromone[depart_node_idx][next_pos] = (1 - self.options.phi) * pheromone[depart_node_idx][next_pos] + self.options.phi * self.options.tau_0
                # Update nodes with new position of ants
                nodes = next_step_nodes
                # Check if ant has completed the route
                if ants[0].complete():
                    break
            # Complete the round trip for each ant
            for ant in ants:
                ant.move(ant.path[0], self.distance_matrix[ant.path[-1]][ant.path[0]])
            # The shortest route of this iteration
            best_ant = min(ants, key=lambda ant: ant.move_distance)
            if best_ant.move_distance < self.best_so_far_length:
                self.best_so_far_length = best_ant.move_distance
                self.best_so_far_path = best_ant.path
            print(f"Best route of iteration {i} is {best_ant.path} with distance {best_ant.move_distance}, bsf distance = {self.best_so_far_length}, bsf path = {self.best_so_far_path}")
            
            # Global update pheromone
            # Only the best ant can update pheromone
            for i in range(len(self.best_so_far_path) - 1):
                pheromone[self.best_so_far_path[i]][self.best_so_far_path[i + 1]] = (1 - self.options.rho) * pheromone[self.best_so_far_path[i]][self.best_so_far_path[i + 1]] + self.options.rho * 1 / best_ant.move_distance
        
            # Reset ant for new iteration (release memory)
            for ant in ants:
                ant.release_memory()
        self.total_iter = self.options.max_iter
        return self.best_so_far_path, self.best_so_far_length
