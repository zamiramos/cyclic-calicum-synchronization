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

def get_disjoint_neighbours(p_src_index, p_dst_index, tp_indices):
    src_neighbor_indices = tp_indices[p_src_index]
    dst_neighbor_indices = tp_indices[p_dst_index]
    
    common_neighbours = list(set(src_neighbor_indices) - set(dst_neighbor_indices))
    
    return common_neighbours

def get_common_neighbours(p_src_index, p_dst_index, tp_indices):
    src_neighbor_indices = tp_indices[p_src_index]
    dst_neighbor_indices = tp_indices[p_dst_index]
    
    common_neighbours = list(set(src_neighbor_indices) & set(dst_neighbor_indices))
    
    return common_neighbours

def grangercausalitytests_mem(p_src_index, p_dst_index, intensity_table, maxlag):
    src_indice_data = None
    
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
        gc_magnitude = np.log(np.var(resid_own) / np.var(resid_joint))
        p_value = result[maxlag][0]['ssr_ftest'][1]        
    except Exception as e:
        #print('error cannot analyze: ' + str(e))
        p_value = 1        
            
    return gc_magnitude, p_value

def get_VAR_noise_matrix(signals, olag):
    from statsmodels.tools.tools import add_constant
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.tsatools import lagmat, lagmat2ds
    
    T = signals.shape[0]
    num_signals = signals.shape[1]
        
    # Now we can compute the VAR model with the computed order :
    VAR_resid = np.zeros((T-olag, num_signals))
    VAR_model = {}
    
    for k in range(0, num_signals):
        # Permuting columns to compute VAR :
        signals = np.concatenate((signals[:,k:],signals[:,0:k]),axis = 1)

        if k == num_signals:
            break

        data = lagmat2ds(signals,olag,trim ='both', dropex = 1)
        datajoint = add_constant(data[:, 1:], prepend=False)
        OLS_ = OLS(data[:,0], datajoint).fit()
        VAR_resid[:,k] = OLS_.resid
        VAR_model[k] = OLS_

    # Computing the noise covariance matrix of the full model :
    VAR_noise_matrix = np.cov(VAR_resid.T)
    
    return VAR_noise_matrix, VAR_resid, VAR_model
    
# read more about this anaysis
# references:
# https://www.mdpi.com/2073-8994/11/8/1004/htm
# DOI 10.1007/s00422-015-0665-3 Conditional Granger causality and partitioned Granger causality: differences and similarities (Sheida Malekpour1 · William A. Sethares1)
# survery in GC variants - https://towardsdatascience.com/inferring-causality-in-time-series-data-b8b75fe52c46#4dd7
# https://github.com/syncpy/SyncPy/blob/70f990971a4b4215549559134812c7469c87c88f/src/Methods/DataFromManyPersons/Univariate/Continuous/Linear/ConditionalGrangerCausality.py
# https://mpra.ub.uni-muenchen.de/2962/1/MPRA_paper_2962.pdf
# https://royalsocietypublishing.org/doi/full/10.1098/rsta.2011.0613#d3e2861
def grangercausalitytests_multivariate_mem(p_src_index, p_dst_index, tp_indices, intensity_table, maxlag):
    from scipy import stats
    #extract common neigbours 
    disjoint_neighbours = get_disjoint_neighbours(p_src_index, p_dst_index, tp_indices)
    
    df_num = df1 = maxlag*2
    df2 = maxlag*(len(disjoint_neighbours) + 2)
    
    #estimate if one of the neigbours is influnce both src and dst
    signals = intensity_table[[p_dst_index, p_src_index], :]
    signals = np.transpose(signals)
    
    VAR_noise_matrix_only_condition, VAR_resid_only_condition, VAR_model_only_condition  = get_VAR_noise_matrix(signals, maxlag)
    
    #estimate if one of the neigbours is influnce both src and dst
    signals = intensity_table[[p_dst_index, p_src_index] + disjoint_neighbours, :]
    signals = np.transpose(signals)
    
    VAR_noise_matrix_all, VAR_resid_all, VAR_model_all = get_VAR_noise_matrix(signals, maxlag)
    
    gc_magnitude = np.log(VAR_noise_matrix_only_condition[0,0]/VAR_noise_matrix_all[0, 0])
    
    # Granger Causality test using ssr (F statistic)
    # [(restricted - unrestricted)/(p2 - p1)] / [unrestricted / VAR_model_all[0].df_resid]
    fgc1 = ((VAR_model_only_condition[0].ssr - VAR_model_all[0].ssr) / (df2 - df1)) / (VAR_model_all[0].ssr / VAR_model_all[0].df_resid)
        
    #print(df2)
    #print(VAR_model_all[0].df_resid)
    #print(len(VAR_resid_all[:, 0]))
    #print(len(VAR_resid_all[:, 0]) - df2 - 1)
    
    #degrees of freedom (p2-p1), (n-p1)        
    p_value = stats.f.sf(fgc1, df2 - df1, VAR_model_all[0].df_resid)
    
    #test direct link
    #gc_magnitude_uncondition, p_value_uncondition = grangercausalitytests_mem(p_src_index, p_dst_index, intensity_table, maxlag)
            
    return gc_magnitude, p_value

def check_raw_gc(src_indice, dst_indice, tp_indices, intensity_table, optimal_lags):
    lag = int(optimal_lags[src_indice, dst_indice])  
    
    gc_magnitude, p_value = grangercausalitytests_multivariate_mem(src_indice, dst_indice, tp_indices, intensity_table, maxlag=lag)

    if (gc_magnitude is None) or (p_value is None):       
        gc_magnitude = 0
        p_value = 1
        
    return gc_magnitude, p_value

def get_optimal_lag(p_src_index, neighbor_indices, normalized_cells_response_curve):
    #get the src neighbors    
    number_of_points = len(neighbor_indices)
    
    src_neighbor_indices = neighbor_indices[p_src_index]
    
    optimal_lag_vector = np.zeros((number_of_points))
    
    for p_dst_index in src_neighbor_indices:
        #find the common neighbours
        dst_neighbor_indices = neighbor_indices[p_dst_index]
        disjoint_neighbours = get_disjoint_neighbours(p_src_index, p_dst_index, neighbor_indices)
        
        src_dst_data = normalized_cells_response_curve[[p_src_index ,p_dst_index], :]
        src_dst_data = np.transpose(src_dst_data)
        model = VAR(src_dst_data)
        maxlags=None

        lag_order_results = model.select_order(maxlags=maxlags)

        lags = [lag_order_results.aic, lag_order_results.bic, lag_order_results.fpe, lag_order_results.hqic]        
        
        min_i = np.argmin(lags)

        model = model.fit(maxlags=lags[min_i], ic=None)
        
        if model.test_whiteness(nlags=lags[min_i]).pvalue < 0.05:
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
        
        break
        
    return optimal_lag_vector

def get_collective_optimal_lag(neighbor_indices, normalized_cells_response_curve, tp_indices, custom_filter):
    number_of_points = len(neighbor_indices)
    adj_optimal_lag_matrix = np.zeros((number_of_points, number_of_points))
    
    for src_indice in range(number_of_points):
        if not custom_filter[src_indice]:
            continue
        
        adj_optimal_lag_matrix[src_indice] = get_optimal_lag(src_indice, tp_indices, normalized_cells_response_curve)
    
    return adj_optimal_lag_matrix