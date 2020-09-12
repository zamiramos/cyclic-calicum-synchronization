import numpy as np
import pandas as pd
from module.serializationhelper import save_object, load_object

def get_optimal_lag_exper(p_src_index, src_neighbor_indices, normalized_cells_response_curve):
    from statsmodels.tsa.api import VAR
    
    #get the src neighbors    
    number_of_points = len(src_neighbor_indices)
    
    optimal_lag_vector = dict()
    
    for p_dst_index in src_neighbor_indices:
        src_dst_data = None
        try:
            src_dst_data = normalized_cells_response_curve[[p_src_index, p_dst_index], :]
            src_dst_data = np.transpose(src_dst_data)            
            model = VAR(src_dst_data)        
            maxlags=None

            lag_order_results = model.select_order(maxlags=maxlags)        

            lags = [lag_order_results.aic, lag_order_results.bic, lag_order_results.fpe, lag_order_results.hqic]        

            min_i = np.argmin(lags)        

            model = model.fit(maxlags=lags[min_i], ic=None)

            p_value_whiteness = model.test_whiteness(nlags=lags[min_i]).pvalue

            if p_value_whiteness == float('nan') or p_value_whiteness < 0.05:
                raise ValueError('found autocorrelation in residuals.')

                #i = models[min_i].k_ar + 1
                #while i < 12 * (models[min_i].nobs/100.)**(1./4):                
                #    result_auto_co = model._estimate_var(i,  trend='c')
                #    if result_auto_co.test_whiteness(nlags=i).pvalue > 0.05:                    
                #        break
                #    i += 1            

                #    print 'error order:' + str(models[min_i].k_ar)                
                #    print 'found correlation ' + str(i)

            optimal_lag_vector[p_dst_index] = lags[min_i]
        except:
            print('src index: ' + str(p_src_index) + ' dst index: ' + str(p_dst_index))
            if src_dst_data is not None:
                print(src_dst_data)
            raise
        
    return optimal_lag_vector

def check_raw_gc(src_indice, dst_indice, intensity_table, lag):
    from module.stathelper import grangercausalitytests_mem
    gc_magnitude = 0
    p_value = 0
    gc_magnitude, p_value = grangercausalitytests_mem(src_indice, dst_indice, intensity_table, maxlag=lag, difference=False)

    if (gc_magnitude is None) or (p_value is None):       
        gc_magnitude = 0
        p_value = 0
        
    return gc_magnitude, p_value

def analyze_network_perm(analyze_result_df_perm, cells_indices, pipe_norm_df, get_role):
    analyze_result_df_perm['total_neighbours'] = analyze_result_df_perm.groupby(['source', 'part'])['destination'].transform(lambda x: x.count())
    analyze_result_df_perm['critical_value'] = 0.05 / analyze_result_df_perm['total_neighbours']
    analyze_result_df_perm['significant'] = analyze_result_df_perm['granger_causality_pvalue'] < analyze_result_df_perm['critical_value']
    analyze_result_df_perm['significant'] = analyze_result_df_perm['significant'].astype(int)

    analyze_cell_stats_gc_out_perm = analyze_result_df_perm.groupby(['source', 'part']).agg('mean').unstack().loc[:, ['significant']].rename(columns={'significant':'gc_out'})
    analyze_cell_stats_gc_in_perm = analyze_result_df_perm.groupby(['destination', 'part']).agg('mean').unstack().loc[:, ['significant']].rename(columns={'significant':'gc_in'})    

    analyze_cell_stats_perm = pd.concat([analyze_cell_stats_gc_in_perm, analyze_cell_stats_gc_out_perm], axis=1, sort=False)
    analyze_cell_stats_perm = analyze_cell_stats_perm.fillna(0)
    
    analyze_cell_stats_flat_perm = analyze_cell_stats_perm.stack()
    
    transformed_norm_df_perm = pipe_norm_df.transform(analyze_cell_stats_flat_perm)
    
    analyze_cell_stats_flat_perm['Manual_0.5'] = np.apply_along_axis(lambda row: get_role(row, 0.5), 1, transformed_norm_df_perm)
    
    state_matrix_perm = analyze_cell_stats_flat_perm.unstack()['Manual_' + str(0.5)].transpose()
    unique_elements, counts_elements = np.unique(state_matrix_perm.loc[:, cells_indices].values.flatten(), return_counts=True)
    normalize_counts_elements = counts_elements/sum(counts_elements)
    
    hub_counts_index = np.where(unique_elements == 0)
    indivdual_counts_index = np.where(unique_elements == 1)

    print(counts_elements[hub_counts_index])
    print(counts_elements[indivdual_counts_index])
    
    return counts_elements[hub_counts_index] / counts_elements[indivdual_counts_index]

def get_hub_ind_count_ratio(state_matrix, cells_indices):
    unique_elements, counts_elements = np.unique(state_matrix.loc[:, cells_indices].values.flatten(), return_counts=True)
    normalize_counts_elements = counts_elements/sum(counts_elements)
    hub_counts_index = np.where(unique_elements == 0)
    indivdual_counts_index = np.where(unique_elements == 1)
    return counts_elements[hub_counts_index] / counts_elements[indivdual_counts_index]

def get_hub_ind_count_ratio_perm(cells_indices, tp_indices_1, tp_indices_2, kpss_and_adf_filter, pipe_norm_df, centroids, cells_response_curve_parts, get_role):
    #collect for each hub/individual cells all neighbours in degree of 1 and 2 to avoid collisions
    cells_neighbours = dict()

    for cell_order_i in range(0, len(cells_indices)):
        cells_neighbours[cell_order_i] = tp_indices_1[cells_indices[cell_order_i]] + tp_indices_2[cells_indices[cell_order_i]]

    ## permuate neibhours
    get_new_location_indices = np.zeros((len(cells_indices)), dtype=np.int32)

    #find for each hub cell his new location
    for cell_order_i in range(0, len(cells_indices)):
        indices_pool = list(filter(lambda x: (x not in cells_neighbours[cell_order_i]) and (x not in np.where(kpss_and_adf_filter == False)[0]), range(0, centroids.shape[0])))
        rand_indice = np.random.randint(len(indices_pool))
        get_new_location_indices[cell_order_i] = indices_pool[rand_indice]
    
    #all cells
    all_cells = set(get_new_location_indices).union(set(cells_indices))

    index = 0
    result_df = pd.DataFrame(columns=['source', 'destination', 'part', 'topological_distance', 'optimal_lag', 'granger_causality_mag', 'granger_causality_pvalue'])
    
    print('calculate role')
    #calculate role
    for cell_old_new_location in zip(get_new_location_indices, cells_indices):
        #from module.stathelper import get_optimal_lag

        cell_new_neighbors_indices = tp_indices_1[cell_old_new_location[0]]

        #get for each neighbor the optimal lag
        cell_new_neighbors_optimal_lag = get_optimal_lag_exper(cell_old_new_location[1], cell_new_neighbors_indices, cells_response_curve_parts[0])

        for dst_indice in cell_new_neighbors_indices:
            if not kpss_and_adf_filter[dst_indice]:
                continue

            #check out-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(cell_old_new_location[1], dst_indice, cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])
            result_df.loc[index] = [cell_old_new_location[1], dst_indice, 0, 1, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1

            #check in-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(dst_indice, cell_old_new_location[1], cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])
            result_df.loc[index] = [dst_indice, cell_old_new_location[1], 0, 1, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1

        cell_new_neighbors_indices = tp_indices_2[cell_old_new_location[0]]

        #get for each neighbor the optimal lag
        cell_new_neighbors_optimal_lag = get_optimal_lag_exper(cell_old_new_location[1], cell_new_neighbors_indices, cells_response_curve_parts[0])

        for dst_indice in cell_new_neighbors_indices:
            if not kpss_and_adf_filter[dst_indice]:
                continue

            #check out-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(cell_old_new_location[1], dst_indice, cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])                
            result_df.loc[index] = [cell_old_new_location[1], dst_indice, 0, 2, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1

            #check in-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(dst_indice, cell_old_new_location[1], cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])
            result_df.loc[index] = [dst_indice, cell_old_new_location[1], 0, 2, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1
            
    print('calculate original neighbours with switched cells')
    #calculate original neighbours with switched cells
    for cell_old_new_location in zip(cells_indices, get_new_location_indices):
        #from module.stathelper import get_optimal_lag

        cell_new_neighbors_indices = tp_indices_1[cell_old_new_location[0]]

        #get for each neighbor the optimal lag
        cell_new_neighbors_optimal_lag = get_optimal_lag_exper(cell_old_new_location[1], cell_new_neighbors_indices, cells_response_curve_parts[0])

        for dst_indice in cell_new_neighbors_indices:        
            if not kpss_and_adf_filter[dst_indice]:
                continue

            #check out-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(cell_old_new_location[1], dst_indice, cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])
            result_df.loc[index] = [cell_old_new_location[1], dst_indice, 0, 1, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1

            #check in-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(dst_indice, cell_old_new_location[1], cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])
            result_df.loc[index] = [dst_indice, cell_old_new_location[1], 0, 1, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1

        cell_new_neighbors_indices = tp_indices_2[cell_old_new_location[0]]

        #get for each neighbor the optimal lag
        cell_new_neighbors_optimal_lag = get_optimal_lag_exper(cell_old_new_location[1], cell_new_neighbors_indices, cells_response_curve_parts[0])

        for dst_indice in cell_new_neighbors_indices:
            if not kpss_and_adf_filter[dst_indice]:
                continue

            #check out-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(cell_old_new_location[1], dst_indice, cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])                
            result_df.loc[index] = [cell_old_new_location[1], dst_indice, 0, 2, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1

            #check in-degree causality
            granger_causality_mag, granger_causality_pvalue = check_raw_gc(dst_indice, cell_old_new_location[1], cells_response_curve_parts[0], cell_new_neighbors_optimal_lag[dst_indice])
            result_df.loc[index] = [dst_indice, cell_old_new_location[1], 0, 2, cell_new_neighbors_optimal_lag[dst_indice], granger_causality_mag, granger_causality_pvalue]
            index +=1
    
    return analyze_network_perm(result_df, all_cells, pipe_norm_df, get_role)

class get_hub_ind_count_ratio_perm_function_context:
    def __init__(self, cells_indices, tp_indices_1, tp_indices_2, kpss_and_adf_filter, pipe_norm_df, centroids, cells_response_curve_parts, get_role):
        self.cells_indices = cells_indices
        self.tp_indices_1 = tp_indices_1
        self.tp_indices_2 = tp_indices_2
        self.kpss_and_adf_filter = kpss_and_adf_filter
        self.pipe_norm_df = pipe_norm_df
        self.centroids = centroids
        self.cells_response_curve_parts = cells_response_curve_parts
        self.get_role = get_role

def get_hub_ind_count_ratio_perm_function(get_hub_ind_count_ratio_perm_function_context):
    np.random.seed()
    return get_hub_ind_count_ratio_perm(get_hub_ind_count_ratio_perm_function_context.cells_indices, 
                                        get_hub_ind_count_ratio_perm_function_context.tp_indices_1, 
                                        get_hub_ind_count_ratio_perm_function_context.tp_indices_2, 
                                        get_hub_ind_count_ratio_perm_function_context.kpss_and_adf_filter, 
                                        get_hub_ind_count_ratio_perm_function_context.pipe_norm_df,
                                        get_hub_ind_count_ratio_perm_function_context.centroids,
                                        get_hub_ind_count_ratio_perm_function_context.cells_response_curve_parts,
                                        get_hub_ind_count_ratio_perm_function_context.get_role)

def calc_hub_indv_perm_scores(analyze_cell_stats_flat, neighbor_indices, pipe_norm_df, kpss_and_adf_filter, centroids, cells_response_curve_parts, get_role, base_path):
    # get original hub/individual cells indices
    state_matrix = analyze_cell_stats_flat.unstack()['Manual_' + str(0.5)].transpose()
    cells_state_matrix_loc_indexes = np.where(state_matrix == 0)[1]
    cells_state_matrix_loc_indexes = np.concatenate([cells_state_matrix_loc_indexes, np.where(state_matrix == 1)[1]])
    cells_indices = state_matrix.iloc[:, cells_state_matrix_loc_indexes].columns
    cells_indices = cells_indices.astype(int)
    
    orginal_hub_ind_count_ratio = get_hub_ind_count_ratio(state_matrix, cells_indices)
        
    hub_indv_perm_scores = load_object(base_path + 'hub_indv_perm_scores_df')
    print(hub_indv_perm_scores)
    
    if hub_indv_perm_scores is None:
        from module.networkhelper import build_network
        from module.networkhelper import get_all_indices_with_toplogical_distance_all, get_all_indices_with_toplogical_distance_specific
        from module.stathelper import check_raw_gc, get_collective_optimal_lag
        from modulev2.analyzetools import analyze_network
    
        #calculate neighbor
        ##build network
        simple_network = build_network(neighbor_indices)
        tp_indices_by_td_all = get_all_indices_with_toplogical_distance_all(simple_network, 2, neighbor_indices)

        tp_indices_1 = get_all_indices_with_toplogical_distance_specific(tp_indices_by_td_all, 1, neighbor_indices)
        tp_indices_2 = get_all_indices_with_toplogical_distance_specific(tp_indices_by_td_all, 2, neighbor_indices)

        from multiprocessing import Pool
        from tqdm.notebook import tqdm
        import time
        import multiprocessing

        class NoDaemonProcess(multiprocessing.Process):
            # make 'daemon' attribute always return False
            def _get_daemon(self):
                return False
            def _set_daemon(self, value):
                pass
            daemon = property(_get_daemon, _set_daemon)

        # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
        # because the latter is only a wrapper function, not a proper class.
        class MyPool(multiprocessing.pool.Pool):
            Process = NoDaemonProcess

        def mute():
            sys.stdout = open(os.devnull, 'w')
            if not sys.warnoptions:
                import warnings
                warnings.simplefilter("ignore")
        
        scores = []
        
        from concurrent.futures import ProcessPoolExecutor
        
        
        #pool = ThreadPool(processes=10, maxtasksperchild=1)
        
        pool_size = 100
        '''
        context_array = np.zeros((pool_size), dtype=object)
        
        for i in range(0, pool_size):
            context_array[i] = get_hub_ind_count_ratio_perm_function_context(cells_indices, tp_indices_1, 
                                                                            tp_indices_2, kpss_and_adf_filter, 
                                                                            pipe_norm_df, centroids, 
                                                                            cells_response_curve_parts, get_role)
        ''' 
        
        '''
        with ProcessPoolExecutor(max_workers = 50) as executor:
            results = list(tqdm(executor.map(get_hub_ind_count_ratio_perm_function, context_array), total=len(context_array)))
            
            for result in results:
                scores.append(result[0])
        '''
        
        #for score in tqdm(pool.imap_unordered(get_hub_ind_count_ratio_perm_function, context_array), total=1):
        #    scores.append(score[0])
        #    pass
        
        for i in tqdm(range(pool_size)):
            scores.append(get_hub_ind_count_ratio_perm(cells_indices, 
                                        tp_indices_1, 
                                        tp_indices_2, 
                                        kpss_and_adf_filter, 
                                        pipe_norm_df, 
                                        centroids, 
                                        cells_response_curve_parts, 
                                        get_role)[0])

        hub_indv_perm_scores = scores
        
        save_object(hub_indv_perm_scores, base_path + 'hub_indv_perm_scores_df')
    
    return orginal_hub_ind_count_ratio, hub_indv_perm_scores