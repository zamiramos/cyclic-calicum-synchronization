import parts_analyzetools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def create_color_mapper(number_of_parts, gc_leader_score_for_parts, metric_nodes_neighbors_score, cmap):
    minima = 1
    maxima = 0
    
    for part in range(0, number_of_parts):        
        maxima = np.maximum(maxima, np.max(metric_nodes_neighbors_score(gc_leader_score_for_parts[part])))
        minima = np.minimum(minima, np.min(metric_nodes_neighbors_score(gc_leader_score_for_parts[part])))
    
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    return mapper

def plot_colormap(mapper):
    mapper._A = []
        
    cbar = plt.colorbar(mapper)    

def plot_voronoi(ax, mapper, regions, vertices, vor, points, gc_significant_leader_scores, custom_filter):    
    number_of_regions = len(regions)
    
    for region_index in range(0, number_of_regions):
        region = regions[region_index]
        node_index = np.where(vor.point_region == region_index)[0]
        
        gc_leader_score = 0
        
        if (len(node_index) != 0) and custom_filter[node_index[0]]:
            gc_leader_score = gc_significant_leader_scores[node_index[0]]
                
        polygon = vertices[region]        
        ax.fill(*zip(*polygon), alpha=1, color=mapper.to_rgba(gc_leader_score))
    
    #ax.plot(points[:,0], points[:,1], 'ko')
    ax.set_xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    ax.set_ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

def plot_distance_cycle_maps(number_of_parts, 
                            analyze_gc_result_cell_level,
                            analyze_result_cell_level,
                            kpss_and_adf_filter,                            
                            score_func,
                            metric_nodes_neighbors_score_func,
                            regions,
                            vertices,                            
                            vor,
                            centroids,
                            filename_path):
    
    for distance in range(1, 16):
        fig, axs = plt.subplots(nrows=1, ncols=number_of_parts, figsize=(100, 10))

        gc_leader_score_for_parts = parts_analyzetools.calc_nodes_neighbors_score_for_parts(number_of_parts,
                                                                                    analyze_gc_result_cell_level, 
                                                                                    analyze_result_cell_level, 
                                                                                    kpss_and_adf_filter, 
                                                                                    distance,
                                                                                    score_func)

        mapper = create_color_mapper(number_of_parts, gc_leader_score_for_parts, metric_nodes_neighbors_score_func, cmap=mpl.cm.Blues)

        for part in range(0, number_of_parts):
            data = metric_nodes_neighbors_score_func(gc_leader_score_for_parts[part])
            custom_filter = np.greater(data, np.zeros((len(data))))

            plot_voronoi(axs[part], mapper, regions, vertices, vor, centroids, 
                         data,
                         custom_filter)

        plot_colormap(mapper)

    plt.savefig(filename_path, dpi=200)
    plt.show()

def color_not_significant_red(signficant_threshold):
    def color_not_significant_red_sign(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
        color = 'red' if val >= signficant_threshold else 'black'
        return 'color: %s' % color
    return color_not_significant_red_sign