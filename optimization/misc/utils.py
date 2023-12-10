import gzip
import tsplib95
import pkg_resources
import numpy as np
from enum import Enum

def unzip_and_parse_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        # Read the content of the uncompressed file into a string
        content = gz_file.read()
        return content

class TSPLIB95Category(Enum):
    TSP = 'tsp' # Symmetric TSP
    ATSP = 'atsp'
    HCP = 'hcp'
    CVRP = 'vrp'
    SOP = 'sop'

def tsplib95_get(category: TSPLIB95Category, problem_name: str):
    """
        Ref: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
        The problem set has been copied into the data folder of this package
        if problem_name.tsp.gz exists, read it and return problem
        if problem_name.opt.tour.gz exists, read it and return optimal tour

    Args:
        problem_name (_type_): _description_
    
    Returns:
        (problem, optimal): _description_
    """
    problem_path = pkg_resources.resource_filename('optimization', f"data/problem_sets/TSPLIB95/{category.value}/{problem_name}.{category.value}.gz")
    solution_path = pkg_resources.resource_filename('optimization', f"data/problem_sets/TSPLIB95/{category.value}/{problem_name}.opt.tour.gz")
    problem = None
    optimal_sol = None
    try:
        with gzip.open(problem_path, 'rt', encoding='utf-8') as gz_file:
            content = gz_file.read()
            problem = tsplib95.parse(content)
        with gzip.open(solution_path, 'rt', encoding='utf-8') as gz_file:
            content = gz_file.read()
            optimal_sol = tsplib95.parse(content)
    except FileNotFoundError:
        if problem == None:
            print(f"Error: {problem_name} not found in TSPLIB95 problem set")
            raise FileNotFoundError
        else:
            print(f"NOTE: Official solution for {problem_name} not exist in TSPLIB95 problem set")
    
    return problem, optimal_sol
    
def get_weight_matrix_from_tsplib(problem: tsplib95.models.StandardProblem):
    """
        Get the weight matrix from the tsplib95.models.StandardProblem object

    Args:
        problem (tsplib95.models.StandardProblem): _description_
    
    Returns:
        _type_: _description_
    """
    # Determine the number of nodes
    num_nodes: int = problem.as_name_dict()['dimension']
    if problem.is_explicit(): # weight type == EXPLICIT
        weight_format = problem.edge_weight_format
        weight_data: list = problem.as_name_dict()['edge_weights']
        weight_data_1d = np.array([item for sublist in weight_data for item in sublist])
        if problem.is_complete():
            # Convert weight matrix format to full matrix
            if weight_format == 'FULL_MATRIX':
                # If the format is already full matrix, use it directly
                weight_matrix = weight_data_1d.reshape((num_nodes, num_nodes))
            elif weight_format in ['UPPER_ROW', 'LOWER_COL']:
                # If the format is upper row, convert to a full matrix
                weight_matrix = np.zeros((num_nodes, num_nodes))
                current_idx = 0 # index in 1-d array
                for i in range(num_nodes-1):
                    weight_matrix[i, (i+1):] = weight_data_1d[current_idx : (current_idx + (num_nodes - i - 1))]
                    current_idx += (num_nodes - i - 1)
                # From upper to full matrix
                weight_matrix += weight_matrix.T
            elif weight_format in ['UPPER_DIAG_ROW', 'LOWER_DIAG_COL']:
                # If the format is upper diagonal row, convert to a full matrix
                weight_matrix = np.zeros((num_nodes, num_nodes))
                current_idx = 0
                for i in range(num_nodes):
                    weight_matrix[i, i:] = weight_data_1d[current_idx : (current_idx + (num_nodes - i))]
                    current_idx += (num_nodes - i)
                # From upper to full matrix
                weight_matrix += weight_matrix.T
            elif weight_format in ['UPPER_COL', 'LOWER_ROW']:
                # Lower row and upper column are the same format (mirrored by diagonal)
                weight_matrix = np.zeros((num_nodes, num_nodes))
                current_idx = 0
                for i in range(num_nodes-1):
                    weight_matrix[:(i+1), i+1] = weight_data_1d[current_idx : (current_idx + i + 1)]
                    current_idx += (i + 1)
                # From lower to full matrix
                weight_matrix += weight_matrix.T
            elif weight_format in ['UPPER_DIAG_COL', 'LOWER_DIAG_ROW']:
                # If the format is upper diagonal column, convert to a full matrix
                weight_matrix = np.zeros((num_nodes, num_nodes))
                current_idx = 0
                for i in range(num_nodes):
                    weight_matrix[:(i+1), i] = weight_data_1d[current_idx : (current_idx + i + 1)]
                    current_idx += (i + 1)
                # From upper to full matrix
                weight_matrix += weight_matrix.T
            else: # FUNCTION -> Calculate from node coordinates + edge weight type
                raise NotImplementedError(f"Conversion for '{weight_format}' format is not implemented.")
        else: # Incomplete graph
            # TODO: Should apply/return tabu list
            raise NotImplementedError("Incomplete graph is not supported")
    else: # weight type in [EUC_2D, EUC_3D, MAN_2D, MAN_3D, MAX_2D, MAX_3D, CEIL_2D, GEO, ATT, XRAY1, XRAY2, SPECIAL]
        weight_type = problem.edge_weight_type
        # Calculate the weight matrix from coordinates
        weight_matrix = np.zeros((num_nodes, num_nodes))
        # Create map from node_idx to node_name
        node_list = []
        for node_label in problem.get_nodes():
            node_list.append(node_label)
        # Handle other formats as needed
        if weight_type == 'EUC_2D':
            # Calculate the weight matrix from coordinates
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                    weight_matrix[i, j] = tsplib95.distances.euclidean(problem.node_coords[node_list[i]], problem.node_coords[node_list[j]])
        elif weight_type == 'MAX_2D':
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                    weight_matrix[i, j] = tsplib95.distances.maximum(problem.node_coords[node_list[i]], problem.node_coords[node_list[j]])
        elif weight_type == 'MAN_2D':
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                    weight_matrix[i, j] = tsplib95.distances.manhattan(problem.node_coords[node_list[i]], problem.node_coords[node_list[j]])
        elif weight_type == 'CEIL_2D':
            import math
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                    weight_matrix[i, j] = tsplib95.distances.euclidean(problem.node_coords[node_list[i]], problem.node_coords[node_list[j]], round=math.ceil)
        elif weight_type == 'GEO':
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                    weight_matrix[i, j] = tsplib95.distances.geographical(problem.node_coords[node_list[i]], problem.node_coords[node_list[j]])
        elif weight_type == 'ATT':
            for i in range(len(node_list)):
                for j in range(len(node_list)):
                    weight_matrix[i, j] = tsplib95.distances.pseudo_euclidean(problem.node_coords[node_list[i]], problem.node_coords[node_list[j]])
        else: # EUC_3D, MAX_3D, MAN_3D, XRAY1, XRAY2, SPECIAL
            raise NotImplementedError(f"Conversion for '{weight_type}' weight type is not implemented.")
    return weight_matrix
