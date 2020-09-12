import pandas as pd
import numpy as np
import parts_analyzetools
from scipy.stats import pearsonr

def exact_mc_perm_test(xs, ys, operator, nmc, custom_filter):
    if custom_filter is not None:
        xs = xs[custom_filter]
        ys = ys[custom_filter]
    
    n, k = len(xs), 0.0
    diff = np.abs(operator(xs, ys)[0])
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff <= np.abs(operator(zs[:n],zs[n:])[0])
    return k / float(nmc)

def apply_custom_filter(ts, custom_filter):
    if custom_filter is not None:
        ts = ts[custom_filter]
        
    return ts

def perm_test_seq_cycles(number_of_parts,
                         analyze_gc_result_cell_level,
                         analyze_result_cell_level,
                         kpss_and_adf_filter,
                         score_func,
                         metric_nodes_neighbors_score_func):
    
    perm_test = pd.DataFrame(columns=range(0, number_of_parts - 1))
    cc_score = pd.DataFrame(columns=range(0, number_of_parts - 1))

    for distance in range(1, 16):
        score_for_parts = parts_analyzetools.calc_nodes_neighbors_score_for_parts(number_of_parts,
                                                                       analyze_gc_result_cell_level, 
                                                                       analyze_result_cell_level, 
                                                                       kpss_and_adf_filter, 
                                                                       distance,
                                                                       score_func) 
        perm_test_leader_scores = {}
        parts_cc_scores = {}
        
        for part in range(0, number_of_parts - 1):
            xs = metric_nodes_neighbors_score_func(score_for_parts[part])
            ys = metric_nodes_neighbors_score_func(score_for_parts[part + 1])
            #sign_data = xs + ys
            #custom_filter = np.greater(sign_data, np.zeros((len(sign_data))))
            
            xs = apply_custom_filter(xs, None)
            ys = apply_custom_filter(ys, None)
            
            perm_test_leader_scores[part] = exact_mc_perm_test(xs, ys, pearsonr, 1000, None)     
            
            parts_cc_scores[part] = pearsonr(xs, ys)[0]
        
        perm_test.loc[distance] = perm_test_leader_scores
        cc_score.loc[distance] = parts_cc_scores
    
    return perm_test, cc_score

def perm_test_feature_cycles(number_of_parts,
                            analyze_gc_result_cell_level,
                            analyze_result_cell_level,
                            feature,
                            kpss_and_adf_filter,                            
                            score_func,
                            metric_nodes_neighbors_score_func):
    
    perm_test = pd.DataFrame(columns=range(0, number_of_parts))
    cc_score = pd.DataFrame(columns=range(0, number_of_parts))

    for distance in range(1, 16):
        score_for_parts = parts_analyzetools.calc_nodes_neighbors_score_for_parts(number_of_parts,
                                                                       analyze_gc_result_cell_level, 
                                                                       analyze_result_cell_level, 
                                                                       kpss_and_adf_filter, 
                                                                       distance,
                                                                       score_func) 
        perm_test_leader_scores = {}
        parts_cc_scores = {}
        
        for part in range(0, number_of_parts):
            xs = metric_nodes_neighbors_score_func(score_for_parts[part])
            ys = feature[part]
            
            #sign_data = xs + ys
            #custom_filter = np.greater(sign_data, np.zeros((len(sign_data))))
            
            xs = apply_custom_filter(xs, None)
            ys = apply_custom_filter(ys, None)
            
            perm_test_leader_scores[part] = exact_mc_perm_test(xs, ys, pearsonr, 1000, None)     
            
            parts_cc_scores[part] = pearsonr(xs, ys)[0]
        
        perm_test.loc[distance] = perm_test_leader_scores
        cc_score.loc[distance] = parts_cc_scores
    
    return perm_test, cc_score