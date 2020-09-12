from numba import jit

def count_neighbors(tp_indices):
    count = 0
    number_of_points = len(tp_indices)
    
    for src_indice in range(number_of_points):
        #multiple by 2 because the out-degree edge and the in degree edge
        count += len(tp_indices[src_indice])*2
    
    return count

def divide_frames_into_n_parts(n_parts, cells_response_curve):
    cells_response_curve_parts = {}
    
    number_of_frames = cells_response_curve.shape[1]
    part_len = number_of_frames / n_parts
    
    for i in range(0, n_parts):
        start_index = i*part_len
        end_index = (i+1)*part_len
        cells_response_curve_parts[i] = cells_response_curve[:, start_index: end_index]
    
    return cells_response_curve_parts

def divide_frames_by_indices(indices, cells_response_curve):
    cells_response_curve_parts = {}

    for i in range(0, len(indices) - 1):
        cells_response_curve_parts[i] = cells_response_curve[:, indices[i]:indices[i+1]]        
    
    return cells_response_curve_parts