import networkx as nx
from numba import jit
import numpy as np
from module.stathelper import check_gc_for_index, check_cc_for_index, check_mi_for_index, collect_gc_pvalue

def neighborhood(G, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items() if length == n]

def neighborhood_all(G, node, n):
    return nx.single_source_dijkstra_path_length(G, node)

def get_all_indices_with_toplogical_distance_all(G, max_distance, neighbor_indices):
    neighbors_tp = {}
    for i in range(len(neighbor_indices)):
        neighbors_tp[i] = neighborhood_all(G, i, max_distance)
    
    return neighbors_tp

def get_all_indices_with_toplogical_distance_specific(neighbors_tp_all, distance, neighbor_indices):
    neighbors_tp = {}
    for i in range(len(neighbor_indices)):
        neighbors_tp[i] = [node for node, length in neighbors_tp_all[i].items() if length == distance]
    
    return neighbors_tp

def get_all_indices_with_toplogical_distance(G, distance, neighbor_indices):
    neighbors_tp = {}
    for i in range(len(neighbor_indices)):
        neighbors_tp[i] = neighborhood(G, i, distance)
    
    return neighbors_tp

def build_network(neighbor_indices):
    G = nx.Graph()
    
    for src_indice in range(len(neighbor_indices)):
        for dst_indice in neighbor_indices[src_indice]:
            G.add_edge(src_indice, dst_indice)
    
    return G

def build_gc_adjacency_matrix_tp(neighbor_indices, intensity_table, tp_indices, custom_filter, optimal_lags, critical_value_out, critical_value_in):
    number_of_points = len(neighbor_indices)
    adj_weights_matrix = np.zeros((number_of_points, number_of_points))
    
    #calculate the number of hypothesis
    if critical_value_out is None:
        #Bonferroni p_value correction
        number_of_hypothesis = np.count_nonzero(optimal_lags)
        critical_value_out = 0.05/float(number_of_hypothesis)
    
    if critical_value_in is None:
        #Bonferroni p_value correction
        number_of_hypothesis = np.count_nonzero(optimal_lags)
        critical_value_in = 0.05/float(number_of_hypothesis)
    
    for src_indice in range(number_of_points):
        if not custom_filter[src_indice]:
            continue
            
        adj_weights_matrix[src_indice, :] = check_gc_for_index(src_indice, 
                                                               tp_indices, 
                                                               intensity_table, 
                                                               optimal_lags, 
                                                               critical_value_out, False, 'out')
        
        adj_weights_matrix[:, src_indice] = check_gc_for_index(src_indice, 
                                                               tp_indices, 
                                                               intensity_table, 
                                                               optimal_lags, 
                                                               critical_value_in, False, 'in')
        
    
    return adj_weights_matrix

def collect_gc_pvalue_by_td(neighbor_indices, intensity_table, tp_indices, custom_filter, optimal_lags):
    gc_pvalue_vector_out = []
    gc_pvalue_vector_in = []
    number_of_points = len(neighbor_indices)
    
    for src_indice in range(number_of_points):
        if not custom_filter[src_indice]:
            continue

        p_value_out, p_value_in = collect_gc_pvalue(src_indice, tp_indices, intensity_table, optimal_lags, False)
            
        gc_pvalue_vector_out.extend(p_value_out)
        gc_pvalue_vector_in.extend(p_value_in)
    
    return gc_pvalue_vector_out, gc_pvalue_vector_in


def build_cc_adjacency_matrix_tp(neighbor_indices, intensity_table, tp_indices, custom_filter, optimal_lags):
    number_of_points = len(neighbor_indices)
    adj_weights_matrix = np.zeros((number_of_points, number_of_points))
    
    #calculate the number of hypothesis
    number_of_hypothesis = np.count_nonzero(optimal_lags)
    
    for src_indice in range(number_of_points):
        if not custom_filter[src_indice]:
            continue
        
        #symmetric measure
        result = check_cc_for_index(src_indice, 
                                    tp_indices, 
                                    intensity_table, 
                                    optimal_lags, 
                                    number_of_hypothesis, False)
            
        adj_weights_matrix[src_indice, :] = result
        adj_weights_matrix[:, src_indice] = result
    
    return adj_weights_matrix

def build_mi_adjacency_matrix_tp(neighbor_indices, intensity_table, tp_indices, custom_filter, optimal_lags):
    number_of_points = len(neighbor_indices)
    adj_weights_matrix = np.zeros((number_of_points, number_of_points))
    
    #calculate the number of hypothesis
    number_of_hypothesis = np.count_nonzero(optimal_lags)
    
    for src_indice in range(number_of_points):
        if not custom_filter[src_indice]:
            continue
        
        #symmetric measure
        result = check_mi_for_index(src_indice, 
                                    tp_indices, 
                                    intensity_table, 
                                    optimal_lags, 
                                    number_of_hypothesis, False)
            
        adj_weights_matrix[src_indice, :] = result
        adj_weights_matrix[:, src_indice] = result

    
    return adj_weights_matrix