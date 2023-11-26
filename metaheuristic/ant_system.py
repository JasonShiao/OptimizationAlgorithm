import numpy as np
# The first ant heuristic algorithm

class Ant:
    def __init__(self, n_pos, init_pos):
        self.current_pos = init_pos
        self.n_pos = n_pos
        self.unvisited_pos = np.ones(n_pos, dtype=bool)
        self.unvisited_pos[self.current_pos] = False
        self.path = []
        self.path.append(self.current_pos)
        self.move_distance = 0

    def move(self, heuristic_info: np.ndarray, pheromone: np.ndarray, alpha: float, beta: float):
        """_summary_

        Args:
            heuristic_info (np.ndarray): _description_
            pheromone (np.ndarray): _description_
            alpha (float): _description_
            beta (float): _description_

        Returns:
            (last_pos, new_pos): the route of the movement
        """
        
        #if self.complete:
        #    raise ValueError("All positions have been visited")
        # Calculate transition probability vector (1-d array) with size = n_pos
        p = np.zeros(self.n_pos)
        #for unvisited_pos in self.unvisited_pos:
        for i in range(self.n_pos):
            if self.unvisited_pos[i]:
                p[i] = (pheromone[self.current_pos][i] ** alpha) * (heuristic_info[self.current_pos][i] ** beta)
        # normalize (make sum = 1)
        p = p / np.sum(p)
        # Select next position by roulette wheel
        next_pos = np.random.choice(np.arange(self.n_pos), p=p)
        last_pos = self.current_pos
        self.current_pos = next_pos
        self.path.append(self.current_pos)
        self.unvisited_pos[self.current_pos] = False
        self.move_distance += heuristic_info[last_pos][self.current_pos]
        return (last_pos, self.current_pos)

class AntSystem:
    """ 
        Ant System algorithm proposed by Dorigo in 1992
        
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
    
    def optimize(self, n_ant: int, n_iter: int, heuristic_info: np.ndarray, Q, init_pheromone: np.ndarray, init_pos: np.ndarray):
        """_summary_

        Args:
            heuristic_info (_type_): _description_
            Q (_type_): pheromone constant
            init_pheromone (_type_): initial pheromone state
            init_pos (_type_): initial position of ants
        """
        # Validate input
        if heuristic_info.shape[0] != heuristic_info.shape[1]:
            raise ValueError("Heuristic info must be a square matrix")
        if heuristic_info.shape != init_pheromone.shape:
            raise ValueError("Heuristic info and init_pheromone must have the same shape")
        if init_pos.shape[0] != n_ant:
            raise ValueError("init_pos must have n_ant rows")
        # Initialize pheromone
        pheromone = init_pheromone
        n_pos = heuristic_info.shape[0]
        best_path = []
        best_distance = np.inf
        for i in range(n_iter):
            # Initialize ants
            ants = [Ant(heuristic_info.shape[0], init_pos[i]) for i in range(n_ant)]
            while sum(ants[0].unvisited_pos) > 0:
                # init delta_pheromone for each step
                sum_delta_pheromone = np.zeros(heuristic_info.shape)
                for k in range(n_ant):
                    # Move each ant
                    route_start, routte_dest = ants[k].move(heuristic_info, pheromone, self.alpha, self.beta)
                    # Accumulate delta_pheromone (Ant-Quantity mode)
                    sum_delta_pheromone[route_start][routte_dest] += Q / heuristic_info[route_start][routte_dest]
                # Update pheromone matrix
                pheromone = (1 - self.rho) * pheromone + sum_delta_pheromone
            # The shortest route of this iteration
            best_ant = min(ants, key=lambda ant: ant.move_distance)
            if best_ant.move_distance < best_distance:
                best_distance = best_ant.move_distance
                best_path = best_ant.path
            print(f"Best route of iteration {i} is {best_ant.path} with distance {best_ant.move_distance}, history best distance = {best_distance}")
        return best_path, best_distance