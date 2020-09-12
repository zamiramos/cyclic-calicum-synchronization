
import numpy as np
from module.networkhelper import build_network
from module.networkhelper import get_all_indices_with_toplogical_distance, get_all_indices_with_toplogical_distance_all, get_all_indices_with_toplogical_distance_specific
from module.randomhelper import choice_random_neighbors_global_level, choice_random_neighbors_cell_level
from module.helpers import count_neighbors
from module.stathelper import get_collective_optimal_lag
from module.networkhelper import build_gc_adjacency_matrix_tp, build_cc_adjacency_matrix_tp, build_mi_adjacency_matrix_tp, collect_gc_pvalue_by_td
from multiprocessing import Pool

class create_analyze_network_by_td_context:
    def __init__(self, tp_indices_by_td_all, neighbor_indices, random_mode, max_neighbors, cells_response_curve, custom_filter, tp_distance):
        self.tp_indices_by_td_all = tp_indices_by_td_all
        self.neighbor_indices = neighbor_indices
        self.random_mode = random_mode
        self.max_neighbors = max_neighbors
        self.cells_response_curve = cells_response_curve
        self.custom_filter = custom_filter
        self.tp_distance = tp_distance

def analyze_network_by_td_private(context):
    tp_indices_random = None

    tp_indices = get_all_indices_with_toplogical_distance_specific(context.tp_indices_by_td_all, context.tp_distance, context.neighbor_indices)        
    #choice random neighbors
    if context.random_mode == 'g':
        tp_indices_random = choice_random_neighbors_global_level(tp_indices, count_neighbors(tp_indices), context.max_neighbors)
    elif context.random_mode == 'c':
        tp_indices_random = choice_random_neighbors_cell_level(tp_indices, count_neighbors(tp_indices), context.max_neighbors)
    else:
        raise Exception('random_mode is unknown. please choose global or cell level.')
    
    optimal_lag = get_collective_optimal_lag(context.neighbor_indices, context.cells_response_curve, tp_indices, context.custom_filter)

    return tp_indices, tp_indices_random, optimal_lag


def analyze_network_by_td(neighbor_indices, cells_response_curve, kpss_and_adf_filter, max_tpd, random_mode, max_neighbors):
    '''
    Analyze the network by topological distance.

    Parameters
    ----------
    neighbor_indices : dict
        The key is the node idice and the value are list of neighbor indices
    cells_response_curve : array[#cells, #frames]
        The cell response of each cell

    max_tpd : int
        Max topological distance to estimate
    random_mode : string
        'g' - global mode.
        'c' - cell mode.
    max_neighbors : int
        Max neighbors for each topological distance. 

        In global mode 'g' the max_neighbors is counted globaly.
        In cell mode 'c' the max_neighbors is counted per cell.
        
        If there is more than the max_neighbors then @max_neighbors are chosen uniformly.

    Returns
    -------
    tp_indices_by_td : dict
        The key is the distance and the value is dict which the key are the node idice 
        and the value are list of neighbor indices by topological distance.
    tp_indices_random_by_td : dict
        Same as @tp_indices_by_td but after random choice.
    optimal_lag_by_td: dict
        The key is the distance and the value is the optimal lag for each pairs.
    '''
    #key - topological distance
    #value - dict(vertice index, neighbors indexes)  neighbors of each vertices
    tp_indices_by_td = {}
    tp_indices_random_by_td = {}
    optimal_lag_by_td = {}

    simple_network = build_network(neighbor_indices)
    tp_indices_by_td_all = get_all_indices_with_toplogical_distance_all(simple_network, max_tpd, neighbor_indices)

    pool = Pool(processes=4)
    results = [pool.apply_async(analyze_network_by_td_private, (create_analyze_network_by_td_context(tp_indices_by_td_all,
                            neighbor_indices,
                            random_mode,
                            max_neighbors,
                            cells_response_curve,
                            kpss_and_adf_filter,
                            tp_distance),)) for tp_distance in range(1, max_tpd + 1)]
    
    for td in range(1, max_tpd + 1):
        tp_indices_by_td[td], tp_indices_random_by_td[td], optimal_lag_by_td[td] = results[td-1].get()

    pool.close()

    '''
    for td in range(1, max_tpd + 1):
        #print 'analyze_network_by_td td: ' + str(td) + ' range: 1-' + str(max_tpd)

        tp_indices_by_td[td] = get_all_indices_with_toplogical_distance_specific(tp_indices_by_td_all, td, neighbor_indices)        
        #choice random neighbors
        if random_mode == 'g':
            tp_indices_random_by_td[td] = choice_random_neighbors_global_level(tp_indices_by_td[td], count_neighbors(tp_indices_by_td[td]), max_neighbors)
        elif random_mode == 'c':
            tp_indices_random_by_td[td] = choice_random_neighbors_cell_level(tp_indices_by_td[td], count_neighbors(tp_indices_by_td[td]), max_neighbors)
        else:
            raise Exception('random_mode is unknown. please choose global or cell level.')        

        #Var Analysis Find The Optimal Lag by Information Criterion
        optimal_lag_by_td[td] = get_collective_optimal_lag(neighbor_indices, cells_response_curve, tp_indices_random_by_td[td], kpss_and_adf_filter)
    '''
    
    return tp_indices_by_td, tp_indices_random_by_td, optimal_lag_by_td

def analyze_gc(neighbor_indices, cells_response_curve, tp_indices, kpss_and_adf_filter, optimal_lag, critical_value_out, critical_value_in):
    result = {}

    result['adjacency_matrix'] = build_gc_adjacency_matrix_tp(
        neighbor_indices, 
        cells_response_curve, 
        tp_indices, 
        kpss_and_adf_filter, 
        optimal_lag, 
        critical_value_out, 
        critical_value_in)
    result['num_significant_edges'] = np.count_nonzero(result['adjacency_matrix'])
    result['num_edges'] = count_neighbors(tp_indices)
    result['sum_gc_magn'] = np.sum(result['adjacency_matrix'])    

    return result

def analyze_cc(neighbor_indices, cells_response_curve, tp_indices, kpss_and_adf_filter, optimal_lag):
    result = {}

    result['adjacency_matrix'] = build_cc_adjacency_matrix_tp(neighbor_indices, cells_response_curve, tp_indices, kpss_and_adf_filter, optimal_lag)
    result['num_edges'] = count_neighbors(tp_indices)
    result['sum_cc_magn'] = np.sum(result['adjacency_matrix'])

    return result

def analyze_mi(neighbor_indices, cells_response_curve, tp_indices, kpss_and_adf_filter, optimal_lag):
    result = {}

    result['adjacency_matrix'] = build_mi_adjacency_matrix_tp(neighbor_indices, cells_response_curve, tp_indices, kpss_and_adf_filter, optimal_lag)
    result['num_edges'] = count_neighbors(tp_indices)
    result['sum_mi_magn'] = np.sum(result['adjacency_matrix'])

    return result

class create_analyze_gc_by_td_context:
    def __init__(self, tp_indices, optimal_lag, neighbor_indices, cells_response_curve, custom_filter, critical_values_out, critical_values_in):
        self.tp_indices = tp_indices
        self.optimal_lag = optimal_lag
        self.neighbor_indices = neighbor_indices
        self.cells_response_curve = cells_response_curve        
        self.custom_filter = custom_filter
        self.critical_values_out = critical_values_out
        self.critical_values_in = critical_values_in        

def analyze_gc_by_td_private(context):
    return analyze_gc(context.neighbor_indices, 
                      context.cells_response_curve, 
                      context.tp_indices, 
                      context.custom_filter, 
                      context.optimal_lag, 
                      context.critical_values_out, 
                      context.critical_values_in)

def analyze_gc_by_td(tp_indices_by_td, optimal_lag_by_td, neighbor_indices, cells_response_curve, kpss_and_adf_filter, critical_values_by_td_out, critical_values_by_td_in):
    gc_result_by_td = {}
    max_tpd = len(tp_indices_by_td)

    pool = Pool()
    results = [pool.apply_async(analyze_gc_by_td_private, (create_analyze_gc_by_td_context(tp_indices_by_td[tp_distance],
                            optimal_lag_by_td[tp_distance],
                            neighbor_indices,
                            cells_response_curve,
                            kpss_and_adf_filter,
                            critical_values_by_td_out[tp_distance],
                            critical_values_by_td_in[tp_distance]),)) for tp_distance in range(1, max_tpd + 1)]
    
    for td in range(1, max_tpd + 1):
        gc_result_by_td[td] = results[td-1].get()

    pool.close()    
    
    return gc_result_by_td

def collect_gc_pvalue(tp_indices_by_td, optimal_lag_by_td, neighbor_indices, cells_response_curve, kpss_and_adf_filter):
    gc_pvalue_result_by_td_out = {}
    gc_pvalue_result_by_td_in = {}
    max_tpd = len(tp_indices_by_td)

    for td in range(1, max_tpd + 1):
        gc_pvalue_result_by_td_out[td], gc_pvalue_result_by_td_in[td] = collect_gc_pvalue_by_td(neighbor_indices, cells_response_curve, tp_indices_by_td[td], kpss_and_adf_filter, optimal_lag_by_td[td])        
    
    return gc_pvalue_result_by_td_out, gc_pvalue_result_by_td_in

def analyze_cc_by_td(tp_indices_by_td, optimal_lag_by_td, neighbor_indices, cells_response_curve, kpss_and_adf_filter):
    cc_result_by_td = {}
    max_tpd = len(tp_indices_by_td)

    for td in range(1, max_tpd + 1):  
        cc_result_by_td[td] = analyze_cc(neighbor_indices, cells_response_curve, tp_indices_by_td[td], kpss_and_adf_filter, optimal_lag_by_td[td])
    
    return cc_result_by_td

def analyze_mi_by_td(tp_indices_by_td, optimal_lag_by_td, neighbor_indices, cells_response_curve, kpss_and_adf_filter):
    mi_result_by_td = {}
    max_tpd = len(tp_indices_by_td)

    for td in range(1, max_tpd + 1):  
        mi_result_by_td[td] = analyze_mi(neighbor_indices, cells_response_curve, tp_indices_by_td[td], kpss_and_adf_filter, optimal_lag_by_td[td])
    
    return mi_result_by_td

def analyze_significant(gc_result_by_td):
    max_tpd = len(gc_result_by_td)    
    significant_prob_by_td = np.zeros((max_tpd))

    for td in range(0, max_tpd):
        significant_prob_by_td[td] = gc_result_by_td[td + 1]['num_significant_edges']/float(gc_result_by_td[td + 1]['num_edges'])
    
    return significant_prob_by_td

def analyze_gc_mean_magnitude(gc_result_by_td):
    max_tpd = len(gc_result_by_td)
    significant_prob_by_td = np.zeros((max_tpd))

    for td in range(0, max_tpd):
        significant_prob_by_td[td] = gc_result_by_td[td + 1]['sum_gc_magn']/float(gc_result_by_td[td + 1]['num_edges'])
            
    return significant_prob_by_td

def analyze_cc_mean_magnitude(cc_result_by_td):
    max_tpd = len(cc_result_by_td)
    cc_mean_magnitude_by_td = np.zeros((max_tpd))

    for td in range(0, max_tpd):
        cc_mean_magnitude_by_td[td] = cc_result_by_td[td + 1]['sum_cc_magn']/float(cc_result_by_td[td + 1]['num_edges'])
            
    return cc_mean_magnitude_by_td