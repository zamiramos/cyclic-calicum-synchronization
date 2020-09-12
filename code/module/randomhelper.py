import random
from numba import jit

@jit
def generate_random_neighbors(centroids_num, neighbors_len_dict):
    new_neighbor_indices = {}
    for i in range(centroids_num):
        new_neighbor_indices[i] = [x for x in random.sample(xrange(0, centroids_num), neighbors_len_dict[i]) if x != i]
        
        rand = random.sample(xrange(0, centroids_num), 1)
        while rand == i:
            rand = random.sample(xrange(0, centroids_num), 1)
        
        new_neighbor_indices[i] += rand        
        
    return new_neighbor_indices

@jit
def choice_random_neighbors_global_level(tp_indices, count, quantity): 
    all_indexes = range(0, count)
    selected_indexes = random.sample(all_indexes, k=quantity)
    new_tp_indices = {}
    
    number_of_points = len(tp_indices)
    index = 0
    
    for src_indice in range(number_of_points):
        neighbors = []
        for dst_indice in tp_indices[src_indice]:
            if index in selected_indexes:
                neighbors.append(dst_indice)
            index += 1
        
        new_tp_indices[src_indice] = neighbors
    
    return new_tp_indices

@jit
def choice_random_neighbors_cell_level(tp_indices, count, quantity):
    new_tp_indices = {}
    number_of_points = len(tp_indices)
    
    for src_indice in range(number_of_points):
        neighbors = []
        number_of_neighbors = len(tp_indices[src_indice])
        if number_of_neighbors < quantity:
            neighbors = tp_indices[src_indice]
        else:
            all_indexes = range(0, number_of_neighbors)
            selected_indexes = random.sample(all_indexes, k=quantity)
            
            index = 0
            for dst_indice in tp_indices[src_indice]:
                if index in selected_indexes:
                    neighbors.append(dst_indice)
                index += 1
        
        new_tp_indices[src_indice] = neighbors
    
    return new_tp_indices