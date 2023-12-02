import numpy as np
from enum import Enum
import argparse

# TODO: Apply tau_0 for local pheromone update


class AntSystemMode(Enum):
    AntCycle = 'cycle'
    AntQuantity = 'quantity'
    AntDensity = 'density'
    MinMaxAntSystem = 'minmax'
    AntColonySystem = 'colony'

class AntSystemOptions:
    def __init__(self, mode: AntSystemMode, 
            n_ant: int, max_iter: int, rho: float, alpha: float, beta: float, Q: float = 1, 
            tau_min: float = 0, tau_max: float = np.inf, 
            phi: float = 0.03, tau_0: float = 1, q_0: float = 0.2):
        self.mode = mode   # mode of ant system
        self.n_ant = n_ant # number of ants
        self.max_iter = max_iter # number of iterations
        self.rho = rho     # evaporation rate (for global update, i.e. offline update)
        self.alpha = alpha # weight of pheromone
        self.beta = beta   # weight of heuristic information
        self.Q = Q         # pheromone constant
        self.tau_min = tau_min # min pheromone
        self.tau_max = tau_max # max pheromone
        self.phi = phi     # pheromone decay coefficient (for local update)
        self.tau_0 = tau_0 # initial pheromone
        self.q_0 = q_0     # q_0 for pseudorandom proportional rule

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

class AntSystem:
    """ 
        Ant System algorithm proposed by Dorigo et al. in 1992, An Investigation of some Properties of an "Ant Algorithm"
        
    """
    def __init__(self, options: AntSystemOptions):
        self.options = options
        self.rng = np.random.default_rng()

    
    def move(self, ant: ArtificialAnt, prob_vector: np.ndarray, distance_matrix: np.ndarray):
        if ant.complete():
            raise ValueError("All positions have been visited")
        
        # Apply tabu list
        prob_vector[ant.path] = 0  # Set probabilities for visited positions to 0
        
        if self.options.mode == AntSystemMode.AntColonySystem:
            # get a random number q uniformly distributed over [0, 1] and compare with q_0
            q = self.rng.uniform(0, 1)
            if q <= self.options.q_0: # Case of Exploitation
                next_pos = np.argmax(prob_vector)
                ant.move(next_pos, distance_matrix[ant.current_pos][next_pos])
                return ant.current_pos
        # Case of Exploration
        normalized_prob_vector = prob_vector / np.sum(prob_vector)
        # Select next position by roulette wheel
        #next_pos = np.random.choice(np.arange(ant.n_pos), p=prob_vector)
        # TODO: Use cumulative sum to speed up (roulette wheel selection)
        next_pos = self.rng.choice(np.arange(ant.n_pos), p = normalized_prob_vector)
        ant.move(next_pos, distance_matrix[ant.current_pos][next_pos])
        return ant.current_pos
    
    def optimize(self, distance_matrix: np.ndarray):
        """_summary_

        Args:
            distance_matrix (_type_): _description_
            Q (_type_): pheromone constant
            init_pheromone (_type_): initial pheromone state
            init_pos (_type_): initial position of ants
        """
        # Validate input
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Heuristic info must be a square matrix")
        heuristic_info = 1 / distance_matrix
        np.fill_diagonal(heuristic_info, 0) # Handle division by 0
        n_pos = distance_matrix.shape[0] # number of nodes (positions)
        # Initialize pheromone
        pheromone = np.ones(distance_matrix.shape)
        history_best_path = []
        history_best_distance = np.inf
        ants: list[ArtificialAnt] = []
        nodes = [[] for i in range(n_pos)] # nodes[i][k] is the k-th ant index at node i (i.e. len(nodes[i]) = b_i in original paper)
        # Init pos (node) for each ant
        for i in range(self.options.n_ant):
            # randomly choose the start node
            start_node = np.random.choice(np.arange(n_pos))
            nodes[start_node].append(i)
            # Initialize ant
            ants.append(ArtificialAnt(n_pos, start_node))
        for i in range(self.options.max_iter):
            # Init delta_pheromone for each iteration
            sum_delta_pheromone = np.zeros(heuristic_info.shape)
            while True:
                # Apply state transition rule (random-proportional rule)
                next_step_nodes = [[] for i in range(n_pos)]
                for depart_node_idx in range(len(nodes)):
                    if self.options.mode != AntSystemMode.AntColonySystem:
                        # Calculate transition probability for the node (city) 
                        # NOTE: Tabu list for each ant is applied inside move() function
                        p: np.ndarray = (pheromone[depart_node_idx] ** self.options.alpha) * (heuristic_info[depart_node_idx] ** self.options.beta)
                        for ant_idx in nodes[depart_node_idx]:
                            # Move each ant
                            new_node_idx = self.move(ants[ant_idx], p.copy(), distance_matrix)
                            next_step_nodes[ants[ant_idx].current_pos].append(ant_idx)
                            if self.options.mode == AntSystemMode.AntDensity:
                                # 1. Accumulate delta_pheromone (with Ant-Density mode)
                                sum_delta_pheromone[depart_node_idx][new_node_idx] += self.options.Q
                            elif self.options.mode == AntSystemMode.AntQuantity:
                                sum_delta_pheromone[depart_node_idx][new_node_idx] += self.options.Q / distance_matrix[depart_node_idx][new_node_idx]
                            else:
                                pass # Other modes do not accumulate delta pheromone here
                    else:
                        for ant_idx in nodes[depart_node_idx]:
                            # Calculate probability for each destination node and move ant
                            p: np.ndarray = (pheromone[depart_node_idx] ** self.options.alpha) * (heuristic_info[depart_node_idx] ** self.options.beta)
                            new_node_idx = self.move(ants[ant_idx], p.copy(), distance_matrix)
                            next_step_nodes[ants[ant_idx].current_pos].append(ant_idx)
                            # Local pheromone update: tau_ij = (1 - phi) * tau_ij + phi * tau_0
                            #pheromone[depart_node_idx][new_node_idx] = (1 - self.options.phi) * pheromone[depart_node_idx][new_node_idx] + self.options.phi * self.options.tau_0
                # Update nodes with new position of ants
                nodes = next_step_nodes
                # Check if ant has completed the route
                if ants[0].complete():
                    break
            # Compute the distance of the entire route (L_k):
            #   Already accumulated distance of most paths in the move function, 
            #   only need to add the path from the last node to the start node
            for ant in ants:
                ant.move_distance += distance_matrix[ant.path[-1]][ant.path[0]]
                ant.path.append(ant.path[0])
            # The shortest route of this iteration
            best_ant = min(ants, key=lambda ant: ant.move_distance)
            if best_ant.move_distance < history_best_distance:
                history_best_distance = best_ant.move_distance
                history_best_path = best_ant.path
            print(f"Best route of iteration {i} is {best_ant.path} with distance {best_ant.move_distance}, history best distance = {history_best_distance}, history best path = {history_best_path}")
            
            # Update pheromone offline (global update)
            # For Ant-Cycle mode, accumulate delta pheromone and divide by the distance of the complete route
            if self.options.mode == AntSystemMode.AntCycle:
                for ant in ants:
                    for i in range(len(ant.path) - 1):
                        sum_delta_pheromone[ant.path[i]][ant.path[i + 1]] += self.options.Q / ant.move_distance
            # For MinMaxAntSystem and Ant Colony System, only the best ant can update pheromone
            elif self.options.mode == AntSystemMode.MinMaxAntSystem or self.options.mode == AntSystemMode.AntColonySystem:
                # NOTE: Could use the history best path or iteration best path, we use the history best path here (as the paper does)
                #for i in range(len(best_ant.path) - 1):
                #    sum_delta_pheromone[best_ant.path[i]][best_ant.path[i + 1]] += self.options.Q / best_ant.move_distance
                for i in range(len(history_best_path) - 1):
                    sum_delta_pheromone[history_best_path[i]][history_best_path[i + 1]] += self.options.Q / history_best_distance
            
            if self.options.mode == AntSystemMode.AntColonySystem:
                # tau = (1 - rho) * tau + rho * sum_delta_tau_best
                pheromone = (1 - self.options.rho) * pheromone + self.options.rho * sum_delta_pheromone
            else:
                # tau = (1 - rho) * tau + sum_delta_tau_best
                pheromone = (1 - self.options.rho) * pheromone + sum_delta_pheromone
                if self.options.mode == AntSystemMode.MinMaxAntSystem:
                    pheromone = np.clip(pheromone, self.options.tau_min, self.options.tau_max)

            # Reset ant for new iteration (release memory)
            for ant in ants:
                ant.release_memory()
        return history_best_path, history_best_distance
