from scipy.spatial import distance
import pandas as pd
from parts_analyzetools import calc_tp_indices_matrix
from multiprocessing import Pool, Value
import numpy as np

class create_gc_cc_table_context:
    def __init__(self, centroids, tp_indices, number_of_points, gc_adjacency_matrix, cc_adjacency_matrix, custom_filter, tp_distance):
        self.centroids = centroids
        self.tp_indices = tp_indices
        self.number_of_points = number_of_points
        self.gc_adjacency_matrix = gc_adjacency_matrix
        self.cc_adjacency_matrix = cc_adjacency_matrix
        self.custom_filter = custom_filter
        self.tp_distance = tp_distance

def create_gc_cc_table_distance(context):

    df = pd.DataFrame(columns=['Cell ID', 'TD', 'Cross-Correlation', 'Signficant Edge'])

    index = 0
    
    for src_indice in range(0, context.number_of_points):
        if not context.custom_filter[src_indice]:
            continue
        
        neighbours_count_out = len(context.tp_indices[src_indice])
        
        for dst_indice in context.tp_indices[src_indice]:
            #if not custom_filter[dst_indice]:
            #    continue

            cc = context.cc_adjacency_matrix[src_indice, dst_indice]

            gc_sign_out = context.gc_adjacency_matrix[src_indice, dst_indice] < 0 
            gc_sign_in = context.gc_adjacency_matrix[dst_indice, src_indice] < 0            
            sign_edge_exist = gc_sign_out | gc_sign_in

            df.loc[index] = [src_indice, context.tp_distance, cc, sign_edge_exist]
            index += 1

    return df

def create_gc_cc_table(centroids, analyze_gc_result_cell_level, analyze_result_cell_level, analyze_cc_result_cell_level, custom_filter, part):
        
    distances = len(analyze_result_cell_level[part]['tp_indices_random'])
        
    df = pd.DataFrame(columns=['Cell ID', 'TD', 'Cross-Correlation', 'Signficant Edge'])

    pool = Pool()
    results = [pool.apply_async(create_gc_cc_table_distance, 
    (create_gc_cc_table_context(centroids,
                             analyze_result_cell_level[part]['tp_indices_random'][tp_distance],
                             len(analyze_result_cell_level[part]['tp_indices_random'][tp_distance]),
                             analyze_gc_result_cell_level[part][tp_distance]['adjacency_matrix'],
                             analyze_cc_result_cell_level[part][tp_distance]['adjacency_matrix'],
                             custom_filter,
                             tp_distance),)) for tp_distance in range(1, distances + 1)]

    for p_result in results:
        df_result = p_result.get()
        df = df.append(df_result)     

    pool.close()
    
    return df