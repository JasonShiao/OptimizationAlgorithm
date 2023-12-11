import numpy as np
from enum import Enum
import argparse
import networkx as nx
from bokeh.models import GraphRenderer, MultiLine, Circle, StaticLayoutProvider, LabelSet
from bokeh.models import (BoxSelectTool, HoverTool, NodesAndLinkedEdges, TapTool)
from bokeh.plotting import figure, output_file, show, from_networkx
from bokeh.io import curdoc
from bokeh.layouts import column

# Book by Dorigo: https://web2.qatar.cmu.edu/~gdicaro/15382/additional/aco-book.pdf
# TODO: Apply tau_0 for local pheromone update

# Simple-ACO = S-ACO

# ACO but seems like Ant System??
# * https://github.com/ai-winter/ros_motion_planning/blob/master/src/planner/global_planner/evolutionary_planner/src/aco.cpp
# *


# 2-opt swap: https://en.wikipedia.org/wiki/2-opt
# 3-opt

class AntSystemVariant(Enum):
    AntCycle = 'cycle'
    AntQuantity = 'quantity'
    AntDensity = 'density'
    MaxMinAntSystem = 'maxmin'
    AntColonySystem = 'colony'

class AntSystemOptions:
    def __init__(self, algo_variant: AntSystemVariant, n_ant: int, max_iter: int, tau_0: float, 
                rho: float = 0.5, alpha: float = 1, beta: float = 2, Q: float = 1):
        if rho < 0 or rho > 1:
            raise ValueError("rho must be in [0, 1]")
        if tau_0 == np.inf:
            raise ValueError("tau_0 must be finite")
        self.algo_variant = algo_variant
        self.n_ant = n_ant # number of ants (typically set to the number of pos)
        self.max_iter = max_iter # number of iterations
        self.rho = rho     # evaporation rate (for global update, i.e. offline update)
        self.alpha = alpha # weight of pheromone (typically 1)
        self.beta = beta   # weight of heuristic information (typically 2-5)
        self.Q = Q         # pheromone constant
        self.tau_0 = tau_0 # initial pheromone state (typically rho / C^nn)

class ArtificialAnt:
    def __init__(self, n_pos, init_pos):
        self.current_pos = init_pos
        self.n_pos = n_pos
        self.path = [] # same as tabu list
        self.path.append(self.current_pos)
        self.move_distance: float = 0

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

class AntSystemBase:
    def __init__(self):
        # Initialize random number generator
        self.rng = np.random.default_rng()
        # State variables which could be reset or preserved:
        self.total_iter = 0
        self.best_so_far_path = []
        self.best_so_far_length = np.inf
        self.plot = None
        
    def reset_state(self):
        self.total_iter = 0
        self.best_so_far_path = []
        self.best_so_far_length = np.inf
    
    def move(self, ant: ArtificialAnt, prob_vector: np.ndarray, distance_matrix: np.ndarray):
        raise NotImplementedError("move() function must be implemented in subclass")

    def optimize(self, dim: int, distance_matrix: np.ndarray, options: AntSystemOptions, network_graph: nx.classes.graph.Graph | None = None):
        """TODO: Add obj_function as input argument,
                 if distance matrix is None, then use obj_function with pheromone only

        Args:
            dim (int): _description_
            distance_matrix (np.ndarray): _description_
            options (AntSystemOptions): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError("optimize() function must be implemented in subclass")

class AntSystem(AntSystemBase):
    """ 
        Ant System algorithm proposed by Dorigo et al. in 1992, An Investigation of some Properties of an "Ant Algorithm"
        
        This class is used to solve simple TSP problems (with a distance matrix)
    """
    def __init__(self):
        super().__init__()
    
    def move(self, ant: ArtificialAnt, prob_vector: np.ndarray, distance_matrix: np.ndarray):
        if ant.complete():
            raise ValueError("All positions have been visited")
        
        # Apply tabu list
        prob_vector[ant.path] = 0  # Set probabilities for visited positions to 0
        if np.inf in prob_vector:
            # select the next position directly
            next_pos = np.argmax(prob_vector)
        else:
            normalized_prob_vector = prob_vector / np.sum(prob_vector)
            # Select next position by roulette wheel
            #next_pos = np.random.choice(np.arange(ant.n_pos), p=prob_vector)
            # TODO: Use cumulative sum to speed up (roulette wheel selection)
            next_pos = self.rng.choice(np.arange(ant.n_pos), p = normalized_prob_vector)
        ant.move(next_pos, distance_matrix[ant.current_pos][next_pos])
        return ant.current_pos
    
    def render_plot_nodes(self, network_graph: nx.classes.graph.Graph):
        """_summary_

        Args:
            network_graph (nx.classes.graph.Graph): _description_
        """
        self.plot = figure(width=800, height=800, tools="pan,wheel_zoom", title="Best Path Graph")
        self.plot.add_tools(HoverTool(tooltips="index: @index, info: @info"), TapTool(), BoxSelectTool())
        self.graph_renderer = GraphRenderer()
        self.graph_renderer.node_renderer.data_source.data = dict(
            index=[node_idx for node_idx in network_graph.nodes()],
            x=[data['coord'][0] for node, data in network_graph.nodes(data=True)],
            y=[data['coord'][1] for node, data in network_graph.nodes(data=True)],
        )
        # Set the coordinates of nodes
        pos_coord = [data['coord'] for node, data in network_graph.nodes(data=True)]
        self.graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=dict(zip(network_graph.nodes(), pos_coord)))
        # Add labels to nodes using LabelSet
        labels = LabelSet(x='x', y='y', text='index', level='glyph',
                            source=self.graph_renderer.node_renderer.data_source, text_align='center', text_baseline='middle')
        self.plot.add_layout(labels)
        self.graph_renderer.node_renderer.glyph = Circle(size=18, fill_color="#4287f5")
        self.graph_renderer.node_renderer.selection_glyph = Circle(size=18, fill_color="red")
        self.graph_renderer.node_renderer.hover_glyph = Circle(size=18, fill_color="green")
        self.graph_renderer.edge_renderer.glyph = MultiLine(line_color="black", line_alpha=0.8, line_width=1)
        self.graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color="#32a852", line_width=5)
        self.graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color="#32a852", line_width=5)
        
        self.graph_renderer.selection_policy = NodesAndLinkedEdges()
        self.graph_renderer.inspection_policy = NodesAndLinkedEdges()
        
        self.plot.renderers.append(self.graph_renderer)
        # Show the plot
        output_file('graph.html')
        show(self.plot)
        #input("Press Enter to continue...")

    def render_plot_edges(self, node_map, best_path):
        """_summary_

        Args:
            network_graph (nx.classes.graph.Graph): _description_
        """
        print("Update graph")
        self.graph_renderer.edge_renderer.data_source.data = dict(
            start=[node_map[best_path[i]] for i in range(len(best_path) - 1)],
            end=[node_map[best_path[i + 1]] for i in range(len(best_path) - 1)]
        )
        #print(self.graph_renderer.edge_renderer.data_source.data)
        # Update the plot
        self.plot.renderers.clear()
        curdoc().clear()  # Clear the existing document to avoid interference with Bokeh server sessions
        self.plot.renderers.append(self.graph_renderer)
        show(self.plot)
        #import time
        #time.sleep(0.5)
    
    def optimize(self, dim: int, distance_matrix: np.ndarray, options: AntSystemOptions, network_graph: nx.classes.graph.Graph | None = None):
        """_summary_

        Args:
            options: AntSystemOptions
        """
        # Validate input dim and matrix
        if dim > distance_matrix.shape[0] or dim > distance_matrix.shape[1]:
            raise ValueError("Distance matrix not match with the input dimension")
        self.n_pos = dim # number of nodes (positions) = dimension
        self.distance_matrix = distance_matrix
        np.fill_diagonal(self.distance_matrix, np.inf)
        #print("distance_matrix = ", distance_matrix)
        self.heuristic_info = 1 / distance_matrix
        #print("heuristic_info = ", self.heuristic_info)
        np.fill_diagonal(self.heuristic_info, 0) # Handle division by 0
        # Validate input options
        if options.algo_variant not in [AntSystemVariant.AntCycle, AntSystemVariant.AntQuantity, AntSystemVariant.AntDensity]:
            raise ValueError("Invalid algorithm variant for AS instance")
        self.options = options
        if network_graph is not None:
            self.render_plot_nodes(network_graph)
        else:
            self.plot = None
        # Initialize pheromone with tau_0
        pheromone = np.full(self.distance_matrix.shape, self.options.tau_0)
        ants: list[ArtificialAnt] = []
        nodes = [[] for i in range(self.n_pos)] # nodes[i][k] is the k-th ant index at node i (i.e. len(nodes[i]) = b_i in original paper)
        # Init pos (node) for each ant
        for i in range(self.options.n_ant):
            # randomly choose the start node
            start_node = np.random.choice(np.arange(self.n_pos))
            nodes[start_node].append(i)
            # Initialize ant
            ants.append(ArtificialAnt(self.n_pos, start_node))
        for i in range(self.options.max_iter):
            # Init delta_pheromone for each iteration
            sum_delta_pheromone = np.zeros(self.heuristic_info.shape)
            while True:
                # Apply state transition rule (random-proportional rule)
                next_step_nodes = [[] for i in range(self.n_pos)]
                for depart_node_idx in range(len(nodes)):
                    # Calculate transition probability for the node (city) 
                    # NOTE: Tabu list for each ant is applied inside move() function
                    #print(f"heuristic: {self.heuristic_info[depart_node_idx]}")
                    #print(f"pherom: {pheromone[depart_node_idx]}")
                    p: np.ndarray = (pheromone[depart_node_idx] ** self.options.alpha) * (self.heuristic_info[depart_node_idx] ** self.options.beta)
                    #print(f"p: {p}")
                    for ant_idx in nodes[depart_node_idx]:
                        # Move each ant
                        new_node_idx = self.move(ants[ant_idx], p.copy(), self.distance_matrix)
                        next_step_nodes[ants[ant_idx].current_pos].append(ant_idx)
                        if self.options.algo_variant == AntSystemVariant.AntDensity:
                            # 1. Accumulate delta_pheromone (with Ant-Density mode)
                            sum_delta_pheromone[depart_node_idx][new_node_idx] += self.options.Q
                        elif self.options.algo_variant == AntSystemVariant.AntQuantity:
                            sum_delta_pheromone[depart_node_idx][new_node_idx] += self.options.Q / self.distance_matrix[depart_node_idx][new_node_idx]
                        else:
                            pass # Other modes do not accumulate delta pheromone here
                # Update nodes with new position of ants
                nodes = next_step_nodes
                # Check if ant has completed the route
                if ants[0].complete():
                    break
            # Compute the distance of the entire route (L_k):
            #   Already accumulated distance of most paths in the move function, 
            #   only need to add the path from the last node to the start node
            for ant in ants:
                last_dist_round_trip = self.distance_matrix[ant.path[-1]][ant.path[0]]
                # Simply update the distance value of the initial node in the path
                ant.path.append(ant.path[0])
                ant.move_distance += last_dist_round_trip
            # The shortest route of this iteration
            best_ant = min(ants, key=lambda ant: ant.move_distance)
            if best_ant.move_distance < self.best_so_far_length:
                self.best_so_far_length = best_ant.move_distance
                self.best_so_far_path = best_ant.path
                # Update graph with the new best path
                if self.plot is not None and network_graph is not None:
                    self.render_plot_edges([node_idx for node_idx in network_graph.nodes()], best_ant.path)
            print(f"Best route of iteration {i} is {best_ant.path} with distance {best_ant.move_distance}, bsf distance = {self.best_so_far_length}, bsf path = {self.best_so_far_path}")
            # Update pheromone offline (global update)
            # For Ant-Cycle mode, accumulate delta pheromone and divide by the distance of the complete route
            if self.options.algo_variant == AntSystemVariant.AntCycle:
                for ant in ants:
                    for i in range(len(ant.path) - 1):
                        sum_delta_pheromone[ant.path[i]][ant.path[i + 1]] += self.options.Q / ant.move_distance
            
            # tau = (1 - rho) * tau + sum_delta_tau_best
            pheromone = (1 - self.options.rho) * pheromone + sum_delta_pheromone

            # Reset ant for new iteration (release memory)
            for ant in ants:
                ant.release_memory()
        self.total_iter = self.options.max_iter
        return self.best_so_far_path, self.best_so_far_length
