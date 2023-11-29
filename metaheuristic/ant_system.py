import numpy as np
from enum import Enum
import argparse


class AntMode(Enum):
    AntQuantity = 'quantity'
    AntDensity = 'density'
    AntCycle = 'cycle'

class Ant:
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
    def __init__(self, rho, alpha, beta):
        """_summary_

        Args:
            rho (_type_): evaporation rate
            alpha (_type_): weight of pheromone
            beta (_type_): weight of heuristic information
        """
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng()

    
    def move(self, ant: Ant, prob_vector: np.ndarray, distance_matrix: np.ndarray):
        if ant.complete():
            raise ValueError("All positions have been visited")
        # Apply tabu list
        prob_vector[ant.path] = 0  # Set probabilities for visited positions to 0
        normalized_prob_vector = prob_vector / np.sum(prob_vector)
        # Select next position by roulette wheel
        #next_pos = np.random.choice(np.arange(ant.n_pos), p=prob_vector)
        # TODO: Use cumulative sum to speed up (roulette wheel selection)
        next_pos = self.rng.choice(np.arange(ant.n_pos), p = normalized_prob_vector)
        ant.move(next_pos, distance_matrix[ant.current_pos][next_pos])
        return ant.current_pos
    
    def optimize(self, n_ant: int, n_iter: int, distance_matrix: np.ndarray, Q: float, mode: AntMode = AntMode.AntQuantity):
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
        ants: list[Ant] = []
        nodes = [[] for i in range(n_pos)] # nodes[i][k] is the k-th ant index at node i (i.e. len(nodes[i]) = b_i in original paper)
        # Init pos (node) for each ant
        for i in range(n_ant):
            # randomly choose the start node
            start_node = np.random.choice(np.arange(n_pos))
            nodes[start_node].append(i)
            # Initialize ant
            ants.append(Ant(n_pos, start_node))
        for i in range(n_iter):
            # Init delta_pheromone for each iteration
            sum_delta_pheromone = np.zeros(heuristic_info.shape)
            while True:
                # State transition
                next_step_nodes = [[] for i in range(n_pos)]
                for depart_node_idx in range(len(nodes)):
                    # Calculate transition probability for the node (city) (Not apply tabu list for each ant here yet)
                    p: np.ndarray = (pheromone[depart_node_idx] ** self.alpha) * (heuristic_info[depart_node_idx] ** self.beta)
                    for ant_idx in nodes[depart_node_idx]:
                        # Move each ant
                        new_node_idx = self.move(ants[ant_idx], p.copy(), distance_matrix)
                        next_step_nodes[ants[ant_idx].current_pos].append(ant_idx)
                        if mode == AntMode.AntDensity:
                            # 1. Accumulate delta_pheromone (with Ant-Density mode)
                            sum_delta_pheromone[depart_node_idx][new_node_idx] += Q
                        elif mode == AntMode.AntQuantity:
                            sum_delta_pheromone[depart_node_idx][new_node_idx] += Q / distance_matrix[depart_node_idx][new_node_idx]
                        else:
                            pass # Ant-Cycle mode does not accumulate delta pheromone here
                # Update nodes with new position of ants
                nodes = next_step_nodes
                # Check if ant has completed the route
                if ants[0].complete():
                    break
            # Compute the distance of the route:
            #   Already accumulated distance of most paths in the move function, 
            #   only need to add the path from the last node to the start node
            for ant in ants:
                ant.move_distance += distance_matrix[ant.path[-1]][ant.path[0]]
                ant.path.append(ant.path[0])
            # For Ant-Cycle mode, accumulate delta pheromone and divide by the distance of the complete route
            if mode == AntMode.AntCycle:
                for ant in ants:
                    for i in range(len(ant.path) - 1):
                        sum_delta_pheromone[ant.path[i]][ant.path[i + 1]] += Q / ant.move_distance
            # Update pheromone matrix
            pheromone = (1 - self.rho) * pheromone + sum_delta_pheromone
            # The shortest route of this iteration
            best_ant = min(ants, key=lambda ant: ant.move_distance)
            if best_ant.move_distance < history_best_distance:
                history_best_distance = best_ant.move_distance
                history_best_path = best_ant.path
            print(f"Best route of iteration {i} is {best_ant.path} with distance {best_ant.move_distance}, history best distance = {history_best_distance}, history best path = {history_best_path}")
            # Reset ant for new iteration (release memory)
            for ant in ants:
                ant.release_memory()
        return history_best_path, history_best_distance
