import numpy as np
import pandas as pd
from module.networkhelper import build_network
from module.networkhelper import get_all_indices_with_toplogical_distance_all, get_all_indices_with_toplogical_distance_specific
from multiprocessing import Pool
from scipy.stats import pearsonr

class create_analyze_network_cc_by_td_context:
    def __init__(self, tp_indices_by_td_all, neighbor_indices, cells_response_curve, custom_filter, max_tpd, part):
        self.tp_indices_by_td_all = tp_indices_by_td_all
        self.neighbor_indices = neighbor_indices                
        self.cells_response_curve = cells_response_curve
        self.custom_filter = custom_filter
        self.max_tpd = max_tpd
        self.part = part

def analyze_network_cc_by_part_private(context):
    max_tpd = context.max_tpd
    number_of_points =  len(context.neighbor_indices)

    result_df = pd.DataFrame(columns=['source', 'destination', 'part', 'topological_distance', 'cross_correlation'])
    index = 0

    for td in range(1, max_tpd + 1):
        tp_indices = get_all_indices_with_toplogical_distance_specific(context.tp_indices_by_td_all, td, context.neighbor_indices)        

        for src_indice in range(number_of_points):
            if not context.custom_filter[src_indice]:
                continue

            for dst_indice in tp_indices[src_indice]:
                if not context.custom_filter[dst_indice]:
                    continue

                cross_correlation = pearsonr(context.cells_response_curve[src_indice, :], context.cells_response_curve[dst_indice, :])[0]
                result_df.loc[index] = [src_indice, dst_indice, context.part, td, cross_correlation]
                index +=1

    return result_df


def analyze_network_cc(neighbor_indices, cells_response_curve_parts, kpss_and_adf_filter, max_tpd):
    
    result_df = pd.DataFrame(columns=['source', 'destination', 'part', 'topological_distance', 'cross_correlation'])

    simple_network = build_network(neighbor_indices)
    tp_indices_by_td_all = get_all_indices_with_toplogical_distance_all(simple_network, max_tpd, neighbor_indices)
    number_of_parts = len(cells_response_curve_parts)

    pool = Pool()
    results = [pool.apply_async(analyze_network_cc_by_part_private, (create_analyze_network_cc_by_td_context(tp_indices_by_td_all,
                            neighbor_indices,
                            cells_response_curve_parts[part],
                            kpss_and_adf_filter,
                            max_tpd,
                            part),)) for part in range(0, number_of_parts)]
    
    for p_result in results:
        result_df = result_df.append(p_result.get())

    pool.close()
    
    return result_df