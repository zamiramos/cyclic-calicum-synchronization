import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.signal import correlate
from math import sqrt
import statsmodels.tsa.stattools
import pandas as pd
from numba import jit
from statsmodels.tsa.api import VAR
from scipy.stats import pearsonr

def validate_stationary(normalized_cells_intensity):
    '''
    return p_value values for two tests adf and kpss
    '''
    critcal_values = None
    p_value_results_adf = np.zeros((normalized_cells_intensity.shape[0]))
    p_value_results_kpss = np.zeros((normalized_cells_intensity.shape[0]))
    
    for i,tseries in enumerate(normalized_cells_intensity):
        result_adf = adfuller(tseries)
        result_kpss = kpss(tseries, regression='ct')
        p_value_results_adf[i] = result_adf[1]
        p_value_results_kpss[i] = result_kpss[1]
        
        '''
        if result_kpss[1] < 0.05: # the data has unit root and is not stationary
            print('i: %i ' % i + 'kpss_stat: %f ' % result_kpss[0] + 'p-value: %f ' % result_kpss[1] + 'lags : %f ' % result_kpss[2])
            
        if result_adf[1] > 0.05: # the data has unit root and is not stationary
            print('i: %i ' % i + 'ADF Statistic: %f ' % result_adf[0] + 'p-value: %f ' % result_adf[1] + 'usedlag: %f ' % result_adf[2] + 'nobs: %f ' % result_adf[3])
            
            if critcal_values is None:
                critcal_values = result_adf[4].items()
        '''
    
    '''
    if critcal_values is not None:
        print('Critical Values:')
        for key, value in critcal_values:
            print('\t%s: %.3f' % (key, value))
    '''
    
    return p_value_results_adf, p_value_results_kpss


def cross_correlation(cells_intensity, i, j):
    a = cells_intensity[i]
    b = cells_intensity[j]
    a = a - a.mean()
    b = b - b.mean()
    c = np.correlate(a, b)
    n = sqrt(np.var(a) * np.var(b)) * len(a) # this is the transformation function
    c = np.true_divide(c,n)
    
    return c

def cells_cross_correlation(cells_intensity, custom_filter):
    cross_correlation_length = cells_intensity.shape[1]*2 - 1
    cells_i_j_cc = np.zeros((cells_intensity.shape[0], cells_intensity.shape[0], cross_correlation_length))
    for cell_idx_i in range(cells_intensity.shape[0]):
        for cell_idx_j in range(cells_intensity.shape[0]):
            if custom_filter[cell_idx_i] and custom_filter[cell_idx_j]:
                cells_i_j_cc[cell_idx_i, cell_idx_j, :] = cross_correlation(cells_intensity, cell_idx_i, cell_idx_j)
    
    return cells_i_j_cc

def tpd_cc_avg(cells_i_j_cc, tp_indices):
    cc_avg = np.zeros((cells_i_j_cc.shape[2]))
    
    count = 0
    for src_indice in range(len(tp_indices)):
        for dst_indice in tp_indices[src_indice]:            
            count += 1
            cc_avg += cells_i_j_cc[src_indice, dst_indice]            
    
    cc_avg = cc_avg/count
    
    return cc_avg

def create_data_for_gc_test(p_src_index, p_dst_index, intensity_table):
    src_dst_data = intensity_table[[p_dst_index, p_src_index], :]
    src_dst_data = np.transpose(src_dst_data)
    
    src_dst_data = pd.DataFrame(src_dst_data)
    src_dst_data = src_dst_data.pct_change(periods=1, axis=0).as_matrix()
    
    #return src_dst_data_t
    return src_dst_data[1:, :] #exclude the NAN values

'''
from cachetools import cached

cache={}
'''
#@cached(cache, key=lambda p_src_index, p_dst_index, intensity_table, maxlag, difference: hash((p_src_index, p_dst_index, maxlag, difference)))
def grangercausalitytests_mem(p_src_index, p_dst_index, intensity_table, maxlag, difference):
    src_indice_data = None
    
    if difference == True:
        src_indice_data = create_data_for_gc_test(p_src_index, p_dst_index, intensity_table)
    else:
        '''
        The Null hypothesis for grangercausalitytests is that the time series in the second column, x2, 
        does NOT Granger cause the time series in the first column, x1. 
        Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, 
        taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 
        if the pvalues are below a desired size of the test.
        '''
        src_indice_data = intensity_table[[p_dst_index, p_src_index], :]        
        src_indice_data = np.transpose(src_indice_data)    
        
    result = None
    gc_magnitude = None
    p_value = None
    try:        
        result = statsmodels.tsa.stattools.grangercausalitytests(src_indice_data, maxlag, verbose=False)
        resid_own = result[maxlag][1][0].resid
        resid_joint = result[maxlag][1][1].resid        
        gc_magnitude = np.log(np.var(resid_joint) / np.var(resid_own))
        p_value = result[maxlag][0]['ssr_ftest'][1]        
    except Exception as e:
        p_value = 1        
            
    return gc_magnitude, p_value

def check_gc_for_index(p_src_index, neighbor_indices, intensity_table, optimal_lags, critical_value, difference, direction='out'):    
    #get the src neighbors
    number_of_points = len(neighbor_indices)
    
    src_neighbor_indices = neighbor_indices[p_src_index]
    
    gc_magnitude_vector = np.zeros((number_of_points))
    
    p_value_bc = critical_value
    
    for indice in src_neighbor_indices:
        #gc-manitute  - The log ratio of the prediction error variances for the bivariate and univariate model.
        #the lag estimation is symmetric so no need to check the opposite
        lag = int(optimal_lags[p_src_index, indice])
        gc_magnitude = 0
        p_value = 0

        if direction == 'out':
            gc_magnitude, p_value = grangercausalitytests_mem(p_src_index, indice, intensity_table, maxlag=lag, difference=difference)
        else:
            gc_magnitude, p_value = grangercausalitytests_mem(indice, p_src_index, intensity_table, maxlag=lag, difference=difference)

        if (gc_magnitude is None) or (p_value is None):       
            continue
        
        if (p_value >= p_value_bc):
            continue
            
        gc_magnitude_vector[indice] = gc_magnitude
        
    return gc_magnitude_vector

def check_raw_gc(src_indice, dst_indice, intensity_table, optimal_lags):
    lag = int(optimal_lags[src_indice, dst_indice])    
    gc_magnitude = 0
    p_value = 0
    gc_magnitude, p_value = grangercausalitytests_mem(src_indice, dst_indice, intensity_table, maxlag=lag, difference=False)

    if (gc_magnitude is None) or (p_value is None):       
        gc_magnitude = 0
        p_value = 0
        
    return gc_magnitude, p_value


def collect_gc_pvalue(p_src_index, neighbor_indices, intensity_table, optimal_lags, difference):        
    src_neighbor_indices = neighbor_indices[p_src_index]
    
    gc_pvalue_vector_out = []
    gc_pvalue_vector_in = []
    
    for indice in src_neighbor_indices:
        #gc-manitute  - The log ratio of the prediction error variances for the bivariate and univariate model.
        lag = int(optimal_lags[p_src_index, indice])
        gc_magnitude, p_value = grangercausalitytests_mem(p_src_index, indice, intensity_table, maxlag=lag, difference=difference)        
        if (gc_magnitude is None) or (p_value is None):            
            continue

        gc_pvalue_vector_out.append(p_value)

        gc_magnitude, p_value = grangercausalitytests_mem(indice, p_src_index, intensity_table, maxlag=lag, difference=difference)

        if (gc_magnitude is None) or (p_value is None):            
            continue

        gc_pvalue_vector_in.append(p_value)
        
    return gc_pvalue_vector_out, gc_pvalue_vector_in

'''
def cross_correlation_normalize(a, b, lag):
    #a = (a - np.mean(a)) / np.sqrt(np.std(a))
    #b = (b - np.mean(b)) / np.sqrt(np.std(b))
    c = np.correlate(a, b)[0]
    n = np.sqrt(np.dot(a, a) * np.dot(b, b)) # this is the transformation function
    c = np.true_divide(c,n)
    
    return c
'''

def check_cc_for_index(p_src_index, neighbor_indices, intensity_table, optimal_lags, number_of_hypothesis, difference):    
    #get the src neighbors
    number_of_points = len(neighbor_indices)
    
    src_neighbor_indices = neighbor_indices[p_src_index]
    
    cc_magnitude_vector = np.zeros((number_of_points))
        
    for indice in src_neighbor_indices:        
        cor_score = pearsonr(intensity_table[p_src_index, :], intensity_table[indice, :])[0]   
            
        cc_magnitude_vector[indice] = cor_score
        
    return cc_magnitude_vector

def calc_MI(x, y, bins):
    from scipy.stats import chi2_contingency
    from scipy import ndimage

    c_xy = np.histogram2d(x, y, bins)[0]
    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(c_xy, sigma=1, mode='constant',
                             output=c_xy)

    g, _, _, _ = chi2_contingency(c_xy, lambda_="log-likelihood")

    mi = 0.5 * g / c_xy.sum()
    mi /= np.log(2)
    return mi

def check_mi_for_index(p_src_index, neighbor_indices, intensity_table, optimal_lags, number_of_hypothesis, difference):    
    #get the src neighbors
    number_of_points = len(neighbor_indices)
    
    src_neighbor_indices = neighbor_indices[p_src_index]
    
    mi_magnitude_vector = np.zeros((number_of_points))    
        
    for indice in src_neighbor_indices:        
        cor_score = calc_MI(intensity_table[p_src_index, :], intensity_table[indice, :], 10)
            
        mi_magnitude_vector[indice] = cor_score
        
    return mi_magnitude_vector

def get_optimal_lag(p_src_index, neighbor_indices, normalized_cells_response_curve):
    #get the src neighbors    
    number_of_points = len(neighbor_indices)
    
    src_neighbor_indices = neighbor_indices[p_src_index]
    
    optimal_lag_vector = np.zeros((number_of_points))
    
    for p_dst_index in src_neighbor_indices:
        src_dst_data = normalized_cells_response_curve[[p_src_index, p_dst_index], :]
        src_dst_data = np.transpose(src_dst_data)        
        model = VAR(src_dst_data)        
        maxlags=None

        lag_order_results = model.select_order(maxlags=maxlags)        

        lags = [lag_order_results.aic, lag_order_results.bic, lag_order_results.fpe, lag_order_results.hqic]        
        
        min_i = np.argmin(lags)        

        var_result = model.fit(maxlags=lags[min_i], ic=None)
        
        portmanteau_test = var_result.test_whiteness(lags[min_i]).pvalue
        if portmanteau_test < 0.05:
            raise ValueError('found autocorrelation in residuals.' + str(portmanteau_test))
            '''                        
            i = lags[min_i] + 1
            while i < 12 * (model.nobs/100.)**(1./4):                
                var_result = model.fit(i, ic=None)
                if var_result.test_whiteness(max(10, i + 1)).pvalue >= 0.05:
                    break
                i += 1
                
                #print('error order:' + str(lags[min_i]))
                #print('found correlation ' + str(i))

            optimal_lag_vector[p_dst_index] = i    
        
            else:
            '''
        optimal_lag_vector[p_dst_index] = lags[min_i]
        
    return optimal_lag_vector

def get_collective_optimal_lag(neighbor_indices, normalized_cells_response_curve, tp_indices, custom_filter):
    number_of_points = len(neighbor_indices)
    adj_optimal_lag_matrix = np.zeros((number_of_points, number_of_points))
    
    for src_indice in range(number_of_points):
        if not custom_filter[src_indice]:
            continue
        
        adj_optimal_lag_matrix[src_indice] = get_optimal_lag(src_indice, tp_indices, normalized_cells_response_curve)
    
    return adj_optimal_lag_matrix