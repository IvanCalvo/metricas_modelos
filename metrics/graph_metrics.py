import numpy as np

from pyvane.measure import tortuosity, num_branch_points, total_length
from netrd.distance import JaccardDistance, \
                            PortraitDivergence, \
                            IpsenMikhailov, \
                            NetLSD, \
                            LaplacianSpectral, \
                            DegreeDivergence, \
                            OnionDivergence

distance_list = [JaccardDistance(),
                PortraitDivergence(),
                IpsenMikhailov(),
                NetLSD(),
                LaplacianSpectral(),
                DegreeDivergence(),
                OnionDivergence()]

def calculate_graph_distances(graph_scores, graph_target):
    results = {}

    for distance in distance_list:
        distance_name = (str(distance)).split('.')[2]
        try: 
            results[distance_name] = distance.dist(graph_scores, graph_target)
        except:
            results[distance_name] = np.nan 

    return results

def calculate_bioproperty(graph_scores):
    results = {}
    results["tortuosity"] = tortuosity(graph_scores, 10)
    results["branchs"] = num_branch_points(graph_scores)
    results["length"] = total_length(graph_scores)
    return results