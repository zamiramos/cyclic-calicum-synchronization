import sys
import math
import numpy as np
import pandas as pd

####
#### yaox12
#### copy from https://github.com/yaox12/DensityPeakCluster
#### A python implementation for 'Clustering by fast search and find of density peaks' in science 2014.
####

### AmosZamir - Adding supprort in dataframe


def get_stat(distance_df):    
    return distance_df, len(distance_df) - 1, max(distance_df.values), min(distance_df.values)

def run_function_on_values(distance_df, func):
    for _, row in distance_df.iterrows():
        yield func(row.values[0])

def get_value(distance_df, i, j):
    return distance_df.loc[pd.IndexSlice[i - 1, j - 1], :].values[0]

def auto_select_dc(distance_df, num, max_dis, min_dis):
    '''
    Auto select the dc so that the average number of neighbors is around 1 to 2 percent
    of the total number of points in the data set
    '''
    dc = (max_dis + min_dis) / 2
    
    while True:
        neighbor_percent = sum(run_function_on_values(distance_df, lambda value: 1 if value < dc else 0)) / num ** 2
        if neighbor_percent >= 0.01 and neighbor_percent <= 0.02:
            break
        if neighbor_percent < 0.01:
            min_dis = dc
        elif neighbor_percent > 0.02:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break

    return dc


def local_density(distance_df, num, dc, gauss = True, cutoff = False):
    '''
    Compute all points' local density
    Return: local density vector of points that index from 1
    '''
    assert gauss and cutoff == False and gauss or cutoff == True
    gauss_func = lambda dij, dc : math.exp(- (dij / dc) ** 2)
    cutoff_func = lambda dij, dc : 1 if dij < dc else 0
    func = gauss_func if gauss else cutoff_func
    rho = [-1] + [0] * num
    for i in range(1, num - 1):
        for j in range(i + 1, num):           
            rho[i] += func(get_value(distance_df, i, j), dc) #symmetric
            rho[j] += func(get_value(distance_df, i, j), dc)

    return np.array(rho, np.float32)


def min_distance(distance_df, num, max_dis, rho):
    '''
    Compute all points' min distance to a higher local density point
    Return: min distance vector, nearest neighbor vector
    '''
    sorted_rho_idx = np.argsort(-rho)
    delta = [0.0] + [max_dis] * num
    nearest_neighbor = [0] * (num + 1)
    delta[sorted_rho_idx[0]] = -1.0
    for i in range(1, num):
        idx_i = sorted_rho_idx[i]
        for j  in range(0, i):
            idx_j = sorted_rho_idx[j]
            if get_value(distance_df, idx_i, idx_j) < delta[idx_i]:
                delta[idx_i] = get_value(distance_df, idx_i, idx_j)
                nearest_neighbor[idx_i] = idx_j

    delta[sorted_rho_idx[0]] = max(delta)
    return np.array(delta, np.float32), np.array(nearest_neighbor, np.int)


class DensityPeakCluster(object):

    def __init__(self):
        self.distance_df = None
        self.rho = None
        self.delta = None
        self.nearest_neighbor = None
        self.num = None
        self.dc = None
        self.core = None


    def density_and_distance(self, distance_df, dc = None):
        distance_df, num, max_dis, min_dis = get_stat(distance_df)
        if dc == None:
            dc = auto_select_dc(distance_df, num, max_dis, min_dis)
        rho = local_density(distance_df, num, dc)
        delta, nearest_neighbor = min_distance(distance_df, num, max_dis, rho)
        
        self.distance_df = distance_df
        self.rho = rho
        self.delta = delta
        self.nearest_neighbor = nearest_neighbor
        self.num = num
        self.dc = dc

        return rho, delta

    def cluster(self, density_threshold, distance_threshold, dc = None):
        if self.distance_df == None:
            print('Please run density_and_distance first.')
            exit(0)
        distance = self.distance_df
        rho = self.rho
        delta = self.delta
        nearest_neighbor = self.nearest_neighbor
        num = self.num
        dc = self.dc

        print('Find the center.')
        cluster = [-1] * (num + 1)
        center = []
        for i in range(1, num + 1):
            if rho[i] >= density_threshold and delta[i] >= distance_threshold:
                center.append(i)
                cluster[i] = i
        
        print('Assignation begings.')
        #assignation
        sorted_rho_idx = np.argsort(-rho)
        for i in range(num):
            idx = sorted_rho_idx[i]
            if idx in center:
                continue
            cluster[idx] = cluster[nearest_neighbor[idx]]

        print('Halo and core.')
        '''
        halo: points belong to halo of a cluster
        core: points belong to core of a cluster, -1 otherwise
        '''
        halo = cluster[:]
        core = [-1] * (num + 1)
        if len(center) > 1:
            rho_b = [0.0] * (num + 1)
            for i in range(1, num):
                for j in range(i + 1, num + 1):
                    if cluster[i] != cluster[j] and get_data(distance, i, j) < dc:
                        rho_avg = (rho[i] + rho[j]) / 2
                        rho_b[cluster[i]] = max(rho_b[cluster[i]], rho_avg)
                        rho_b[cluster[j]] = max(rho_b[cluster[j]], rho_avg)

            for i in range(1, num + 1):
                if rho[i] > rho_b[cluster[i]]:
                    halo[i] = -1
                    core[i] = cluster[i]

        for i in range(len(center)):
            n_ele, n_halo = 0, 0
            for j in range(1, num + 1):
                if cluster[j] == center[i]:
                    n_ele += 1
                if halo[j] == center[i]:
                    n_halo += 1
            n_core = n_ele - n_halo
            print("Cluster %d: Center: %d, Element: %d, Core: %d, Halo: %d\n" % (i + 1, center[i], n_ele, n_core, n_halo))

        self.core = core           

        return rho, delta, nearest_neighbor