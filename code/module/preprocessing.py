import scipy.io
import numpy as np

from statsmodels.nonparametric.smoothers_lowess import lowess

def smooth(cells_intensity, frac=0.01):
    '''
    smooth function based on lowess function
    '''
    smooth_intensity = np.zeros((cells_intensity.shape[0], cells_intensity.shape[1]))    
    for i in range(cells_intensity.shape[0]):    
        smooth_intensity[i,:] = lowess(cells_intensity[i], range(cells_intensity.shape[1]), frac)[:, 1]
    
    return smooth_intensity

def five_point_stencil(cell_intensity):
    '''
    The first derivative of a function f of a real variable at a point x can be approximated using a five-point stencil
    '''
    return np.convolve(cell_intensity, [1, -8, 8, -1], mode='same')/12

def first_derive_cells_intensity(cells_intensity):
    '''
    based on five_point_stencil function
    '''
    first_derive_cells_intensity = np.zeros((cells_intensity.shape[0], cells_intensity.shape[1]))
    for cell_idx,cell_intensity in enumerate(cells_intensity):
        first_derive = five_point_stencil(cell_intensity)        
        first_derive_cells_intensity[cell_idx, :] = first_derive - np.mean(first_derive)
    
    return first_derive_cells_intensity[:, 2:-3]