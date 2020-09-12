import matplotlib as mpl
import numpy as np
from numba import jit

def centrality_metric_nodes_neighbors_score(gc_leader_score):
    return gc_leader_score['out'] - gc_leader_score['in']

def calc_gc_magnitude_score(src_adjacency_array, num_neighbors):
    if num_neighbors == 0:
        return 0
    
    return -np.sum(src_adjacency_array)/float(num_neighbors)

def calc_gc_significant_score(src_adjacency_array, num_neighbors):
    if num_neighbors == 0:
        return 0
    
    return np.count_nonzero(src_adjacency_array)/float(num_neighbors)

@jit
def calc_tp_indices_matrix(tp_indices, custom_filter):
    number_of_points = len(tp_indices)
    
    tp_matrix_indices = np.zeros((number_of_points, number_of_points))
    
    for src_indice in range(0, number_of_points):
        if not custom_filter[src_indice]:
            continue
        for dst_indice in tp_indices[src_indice]:
            tp_matrix_indices[src_indice, dst_indice] = 1
            tp_matrix_indices[dst_indice, src_indice] = 1
    
    return tp_matrix_indices
    
    
def calc_nodes_neighbors_score(gc_adjacency_matrix, tp_indices, custom_filter, score_neighbors_func):
    number_of_points = len(tp_indices)
    
    gc_significant_leader_scores = {}
    gc_significant_leader_scores['out'] = np.zeros((number_of_points))
    gc_significant_leader_scores['in'] = np.zeros((number_of_points))
    
    #tp_indices_matrix contains simple binary array of 1 for edge and 0 for non-edge
    tp_indices_matrix = calc_tp_indices_matrix(tp_indices, custom_filter)
    
    for src_indice in range(0, number_of_points):
        if not custom_filter[src_indice]:
            continue        
        
        gc_significant_leader_scores['out'][src_indice] = score_neighbors_func(
            gc_adjacency_matrix[src_indice, :], 
            np.sum(tp_indices_matrix[src_indice, :]))
        gc_significant_leader_scores['in'][src_indice] = score_neighbors_func(
            gc_adjacency_matrix[:, src_indice], 
            np.sum(tp_indices_matrix[:, src_indice]))
    
    gc_significant_leader_scores['in'] = np.array(gc_significant_leader_scores['in'])
    gc_significant_leader_scores['out'] = np.array(gc_significant_leader_scores['out'])
    
    return gc_significant_leader_scores

def out_metric_nodes_neighbors_score(gc_signifcant_leader_score):
    return gc_signifcant_leader_score['out']

def in_metric_nodes_neighbors_score(gc_signifcant_leader_score):
    return gc_signifcant_leader_score['in']

def create_color_mapper(number_of_parts, gc_signifcant_leader_score_for_parts, metric_nodes_neighbors_score):
    minima = 1
    maxima = 0
    
    for part in range(0, number_of_parts):        
        maxima = np.maximum(maxima, np.max(metric_nodes_neighbors_score(gc_signifcant_leader_score_for_parts[part])))
        minima = np.minimum(minima, np.min(metric_nodes_neighbors_score(gc_signifcant_leader_score_for_parts[part])))
    
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    
    return mapper

def calc_nodes_neighbors_score_for_parts(number_of_parts, 
                                         analyze_gc_result_cell_level, 
                                         analyze_result_cell_level, 
                                         kpss_and_adf_filter, 
                                         distance, 
                                         score_neighbors_func):
    gc_signifcant_leader_scores = {}
    for part in range(0, number_of_parts):
        gc_signifcant_leader_scores[part] = calc_nodes_neighbors_score(analyze_gc_result_cell_level[part][distance]['adjacency_matrix'], 
                                             analyze_result_cell_level[part]['tp_indices_random'][distance],
                                             kpss_and_adf_filter, 
                                             score_neighbors_func)
    
    return gc_signifcant_leader_scores