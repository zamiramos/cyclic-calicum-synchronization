import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay

##voronoi diagram

'''
in order to calculate the bounding box area the following steps are needed:
1. hull = ConvexHull(centroids)
2. rect_vert = bounding_box(centroids[hull.vertices])
3. area_of_rectangle(rect_vert[1][0]-rect_vert[0][0], rect_vert[2][1]-rect_vert[1][1])

'''

def area_of_rectangle(width, length):
    area = float(width) * float(length)
    print('The Area of the Rectangle is {:}'. format(area))
    return area

def bounding_box(coords):
    min_x = coords[0][0] # start with something much higher than expected min
    min_y = coords[0][1]
    max_x = coords[0][0] # start with something much lower than expected max
    max_y = coords[0][1]

    for item in coords:
        if item[0] < min_x:
            min_x = item[0]

        if item[0] > max_x:
            max_x = item[0]

        if item[1] < min_y:
            min_y = item[1]

        if item[1] > max_y:
            max_y = item[1]
    
    return [(min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y)]

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def find_neighbors(tess, points):
    '''
    Parameters
    ----------
    tess : Delaunay
        Input Delaunay triangulation
    points : array
        The centroids coordinates
    '''
    
    neighbors = {}
    for point in range(points.shape[0]):
        neighbors[point] = []

    for simplex in tess.simplices:
        neighbors[simplex[0]] += [simplex[1],simplex[2]]
        neighbors[simplex[1]] += [simplex[2],simplex[0]]
        neighbors[simplex[2]] += [simplex[0],simplex[1]]
    
    neighbors_set = {}
    
    for (key, value) in neighbors.items():
        neighbors_set[key] = list(set(value))

    return neighbors_set

def split_to_small_bounding_boxes(n, rect_vert):
    total_width = rect_vert[1][0] - rect_vert[0][0]
    total_height = rect_vert[3][1] - rect_vert[1][1]
    
    width_step = total_width / float(n)
    height_step = total_height / float(n)
    
    bb_result = {}
    
    bb_index = 0
    
    for upper_left_x in np.arange(rect_vert[0][0], total_width, width_step):
        for upper_left_y in np.arange(rect_vert[0][1], total_height, height_step):
            bb_coordinates = []
            bb_coordinates.append([upper_left_x, upper_left_y])
            bb_coordinates.append([upper_left_x + width_step, upper_left_y])
            bb_coordinates.append([upper_left_x + width_step, upper_left_y + height_step])
            bb_coordinates.append([upper_left_x, upper_left_y + height_step])
            
            bb_result[bb_index] = bb_coordinates
            
            bb_index += 1
    
    return bb_result

def in_rect(point, rect_vert):
    if point[0] < rect_vert[0][0] or point[0] > rect_vert[1][0]:        
        return False
    
    if point[1] < rect_vert[0][1] or point[1] > rect_vert[2][1]:        
        return False
    
    return True

def filter_edges_in_bb(centroids, edge_weights, rect_vert):
    number_of_points = len(edge_weights)
    filter_edge_weights = edge_weights.copy()
    
    for point_index in range(0, number_of_points):
        if not in_rect(centroids[point_index], rect_vert):
            filter_edge_weights[point_index] =  []
        else:          
            filter_edge_weights[point_index] = [node for node in filter_edge_weights[point_index] if in_rect(centroids[node], rect_vert)]
    
    return filter_edge_weights

def filter_edges_in_bb_by_td(centroids, edge_weights_by_td, rect_vert):
    filter_edge_weights_by_td = {}
    
    for td in range(1, len(edge_weights_by_td) + 1):
        filter_edge_weights_by_td[td] =  filter_edges_in_bb(centroids, edge_weights_by_td[td], rect_vert)
    
    return filter_edge_weights_by_td
