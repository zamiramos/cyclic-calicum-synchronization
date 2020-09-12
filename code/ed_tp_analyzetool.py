from scipy.spatial import distance
import pandas as pd
from parts_analyzetools import calc_tp_indices_matrix
from multiprocessing import Pool, Value
import numpy as np

class create_gc_table_context:
    def __init__(self, centroids, tp_indices, number_of_points, gc_adjacency_matrix, custom_filter, tp_distance):
        self.centroids = centroids
        self.tp_indices = tp_indices
        self.number_of_points = number_of_points
        self.gc_adjacency_matrix = gc_adjacency_matrix
        self.custom_filter = custom_filter
        self.tp_distance = tp_distance        

def create_gc_table_distance(context):

    df_out = pd.DataFrame(columns=['Cell ID', 'Euclidean_Distance', 'Topological_Distance', 'Number of Neighbours', 'GC_Out'])
    df_in = pd.DataFrame(columns=['Cell ID', 'Euclidean_Distance', 'Topological_Distance', 'Number of Neighbours', 'GC_In'])

    #tp_indices_matrix contains simple binary array of 1 for edge and 0 for non-edge
    tp_indices_matrix = calc_tp_indices_matrix(context.tp_indices, context.custom_filter)

    index_out = 0
    index_in = 0
    
    for src_indice in range(0, context.number_of_points):
        if not context.custom_filter[src_indice]:
            continue
        
        neighbours_count_out = len(context.tp_indices[src_indice])
        
        for dst_indice in context.tp_indices[src_indice]:
            #if not custom_filter[dst_indice]:
            #    continue
            
            ed = distance.euclidean(context.centroids[src_indice], context.centroids[dst_indice])
            
            gc_out = context.gc_adjacency_matrix[src_indice, dst_indice]            

            df_out.loc[index_out] = [src_indice, ed, context.tp_distance, neighbours_count_out, gc_out]
            index_out += 1
        
        neighbours_in = np.nonzero(tp_indices_matrix[:, src_indice])[0]
            
        neighbours_count_in = len(neighbours_in)
        
        for dst_indice in neighbours_in:
            ed = distance.euclidean(context.centroids[src_indice], context.centroids[dst_indice])                
            
            gc_in = context.gc_adjacency_matrix[dst_indice, src_indice]                    
                            
            df_in.loc[index_in] = [src_indice, ed, context.tp_distance, neighbours_count_in, gc_in]
            
            index_in += 1
    

    return df_out, df_in

def create_gc_table(centroids, analyze_gc_result_cell_level, analyze_result_cell_level, custom_filter, part):
        
    distances = len(analyze_result_cell_level[part]['tp_indices_random'])
        
    df_out = pd.DataFrame(columns=['Cell ID', 'Euclidean_Distance', 'Topological_Distance', 'Number of Neighbours', 'GC_Out'])   
    df_in = pd.DataFrame(columns=['Cell ID', 'Euclidean_Distance', 'Topological_Distance', 'Number of Neighbours', 'GC_In'])  

    pool = Pool()
    results = [pool.apply_async(create_gc_table_distance, 
    (create_gc_table_context(centroids,
                             analyze_result_cell_level[part]['tp_indices_random'][tp_distance],
                             len(analyze_result_cell_level[part]['tp_indices_random'][tp_distance]),
                             analyze_gc_result_cell_level[part][tp_distance]['adjacency_matrix'],
                             custom_filter,
                             tp_distance),)) for tp_distance in range(1, distances + 1)]

    for p_result in results:
        df_result_out, df_result_in = p_result.get()
        df_out = df_out.append(df_result_out)
        df_in = df_in.append(df_result_in)

    pool.close()
    
    return df_out, df_in