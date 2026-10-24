# Some usefull metrics to evaluate morphology conservation on the segmentation of blood vessels
# All functions compare pytorch tensors

import torch
import numpy as np
import networkx as nx
import scipy.ndimage as ndi
from pyvane.image import Image
import skimage.morphology as morph
from pyvane.graph import create_graph
from typing import Tuple

from torchtrainer.util.post_processing import logits_to_preds

def cl_dice(gt: torch.Tensor, predicted: torch.Tensor) -> float:
    """
    Calculates the centerline Dice between batches of 2D binary images.
    https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html

    Parameters
    ------------------------------------------------
    gt: torch.Tensor
        A binary torch.Tensor with shape [/8].
    
    predicted: torch.Tensor
        A binary torch.Tensor with shape [B, H, W].

    Returns
    ------------------------------------------------
    cl_dice: float
        Average clDice of the whole batch.
    """

    np_gt = gt.detach().cpu().numpy()
    np_predicted = predicted.detach().cpu().numpy()

    cl_dices = []
    for Vl, Vp in zip(np_gt, np_predicted):
        # since the batched images do not correspond to a volume, we need to perform several
        # 2D skeletonizations
        Sl = morph.skeletonize(Vl)
        Sp = morph.skeletonize(Vp)

        # fraction of predicted skeleton that lies within the ground truth volume
        sum_sp = np.sum(Sp)
        if sum_sp == 0:
            tprec = 0
        else:
            tprec = np.sum(np.logical_and(Sp, Vl)) / np.sum(Sp)
        # fraction of the ground truth skeleton that lies within the predicted volume
        sum_sl = np.sum(Sl)
        if sum_sl == 0:
            tsens = 0
        else:
            tsens = np.sum(np.logical_and(Sl, Vp)) / np.sum(Sl)

        den = tprec + tsens
        if den == 0:
            cl_dices.append(0)
        else:
            cl_dices.append(2 * (tprec*tsens) / den)

    return np.mean(cl_dices).item()

def betti_numbers(inputs: torch.Tensor) -> tuple[int, int]:
    """
    Calculates the 0- and 1-dimensional homology of a batch of 2D binary images.
    0-dimensional homology is defined as the number of connected components of white pixels
    (with value > 0). 1-dimensional homology is defined as the number of connected clusters of
    black pixels (which are surrounded by white pixels). This function also considers a black
    cluster enclosed by border pixels to be part of a 1-dimensional component.

    Parameters
    ------------------------------------------------
    inputs: torch.Tensor
        A binary torch.Tensor with shape [B, H, W].

    Returns
    ------------------------------------------------
    beta_0: int
        0-dimensional homology value.
    
    beta_1: int
        1-dimensional homology value.
    """

    beta_0 = 0
    beta_1 = 0
    for image in inputs:
        binary = (image > 0).detach().cpu().numpy()
        _, n_components = morph.label(binary, return_num=True)
        # number of connected components of white pixels
        beta_0 += n_components
    
        binary = (image == 0).detach().cpu().numpy()
        _, n_components = morph.label(binary, return_num=True)
        # number of connected components of black pixels
        beta_1 += n_components

    return beta_0, beta_1

def ccq(gt: torch.Tensor, predicted: torch.Tensor, distance: int = 3) -> tuple[float, float, float]:
    """
    CCQ stands for correctness, completeness and quality. They are similar to precision, recall,
    and F1-score, but the predicted foreground pixels are counted as true positives if they are
    at least `distance` pixels away from a ground truth true positive.

    https://www.unibw.de/visual-computing/publikationen-vortraege/publikationen-1/pdf-dateien-publikationen/ieee-1997.pdf
    """
    
    np_gt = gt.detach().cpu().numpy() > 0
    np_predicted = predicted.detach().cpu().numpy() > 0

    corr_vals = []
    comp_vals = []
    quality_vals = []
    
    for input, target in zip(np_predicted, np_gt):
        dtransf = ndi.distance_transform_edt(1-target, return_distances=True)
        # here, we expand the labels to consider pixels close to the foreground as foreground
        exp_label = (dtransf <= distance)

        tp = np.sum((input) & (exp_label))
        fp = np.sum((input) & (~exp_label))
        fn = np.sum((~input) & (exp_label))
        
        den = tp + fp
        if den == 0:
            correctness = 0
        else:
            correctness = tp / den

        den = tp + fn
        if den == 0:
            completeness = 0
        else:
            completeness = tp / den
        
        den = correctness + completeness
        if den == 0:
            quality = 0
        else:
            quality = 2 * (correctness * completeness) / den

        corr_vals.append(correctness)
        comp_vals.append(completeness)
        quality_vals.append(quality)

    return np.mean(corr_vals).item(), np.mean(comp_vals).item(), np.mean(quality_vals).item()

def ccq_new(gt: torch.Tensor, predicted: torch.Tensor, distance: int = 3) -> tuple[float, float, float]:
    """
    CCQ stands for correctness, completeness and quality. They are similar to precision, recall,
    and F1-score, but the predicted foreground pixels are counted as true positives if they are
    at least `distance` pixels away from a ground truth true positive.
    https://www.unibw.de/visual-computing/publikationen-vortraege/publikationen-1/pdf-dateien-publikationen/ieee-1997.pdf
    """

    np_gt = gt.detach().cpu().numpy() > 0
    np_predicted = predicted.detach().cpu().numpy() > 0
    corr_vals = []
    comp_vals = []
    quality_vals = []

    for input, target in zip(np_predicted, np_gt):
        dtransf = ndi.distance_transform_edt(1 - target, return_distances=True)
        exp_label = (dtransf <= distance)

        tp = np.sum(( input) & ( exp_label))
        fp = np.sum(( input) & (~exp_label))
        fn = np.sum((~input) & ( target))     # ← original target, not exp_label

        den = tp + fp
        correctness  = tp / den if den > 0 else 0.0

        den = tp + fn
        completeness = tp / den if den > 0 else 0.0

        den = correctness + completeness
        quality = 2 * (correctness * completeness) / den if den > 0 else 0.0

        corr_vals.append(correctness)
        comp_vals.append(completeness)
        quality_vals.append(quality)

    return np.mean(corr_vals).item(), np.mean(comp_vals).item(), np.mean(quality_vals).item()

def _add_weights(graph: nx.MultiGraph) -> None:
    """Calculates the path length of each edge in `graph` and adds it as an edge attribute."""

    weights = {}
    for n1, n2, key in graph.edges:
        distance = 0.
        path = graph.get_edge_data(n1, n2, key)['path']
        for i in range(1, len(path)):
            pix1 = path[i-1]
            pix2 = path[i]
            distance += np.sqrt((pix2[0]-pix1[0])**2 + (pix2[1]-pix1[1])**2)
        weights[(n1, n2, key)] = distance
    
    nx.set_edge_attributes(graph, weights, 'weight')

def _get_all_pairs_of_nodes(nodes: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            pairs.append((nodes[i], nodes[j]))
    
    return pairs

def apls(gt: torch.Tensor, predicted: torch.Tensor, node_threshold: int = 12) -> float:
    """APLS: Average Path-Length Similarity. Calculates the differences between the shortest
    distances between pairs of nodes of the ground truth and predicted graphs.
    
    https://arxiv.org/abs/1807.01232
    """
    
    np_gt = gt.detach().cpu().numpy()
    np_predicted = predicted.detach().cpu().numpy()

    # if images are not normalized
    if np.max(np_gt) > 1:
        np_gt = np_gt // np.max(np_gt)
    if np.max(np_predicted) > 1:
        np_predicted = np_predicted // np.max(np_predicted)

    C_values = []
    for input, target in zip(np_predicted, np_gt):
        # generate graphs
        gt_graph = create_graph(Image(morph.skeletonize(target)))
        predicted_graph = create_graph(Image(morph.skeletonize(input)))
    
        # calculate the path lenghts of each edge
        _add_weights(gt_graph)
        _add_weights(predicted_graph)

        # choose which nodes will be used in the calculation
        if gt_graph.number_of_nodes() <= node_threshold:
            nodes_to_calculate = list(gt_graph.nodes)
        else:
            all_nodes = list(gt_graph.nodes)
            nodes_to_calculate = np.random.choice(all_nodes, size=node_threshold, replace=True)

        all_pairs = _get_all_pairs_of_nodes(nodes_to_calculate)
        for pair in all_pairs:
            start, end = pair
            # match the ground truth node to a node in predicted_graph
            gt_start = gt_graph.nodes[start]['center']
            gt_end = gt_graph.nodes[end]['center']
            
            distances = []
            for nodes in predicted_graph.nodes:
                pred_pos = predicted_graph.nodes[nodes]['center']
                distances.append(np.sqrt((pred_pos[0]-gt_start[0])**2 + (pred_pos[1]-gt_start[1])**2))
            smallest_idx = np.argmin(distances)
            matched_start = list(predicted_graph.nodes)[smallest_idx]

            distances = []
            for nodes in predicted_graph.nodes:
                pred_pos = predicted_graph.nodes[nodes]['center']
                distances.append(np.sqrt((pred_pos[0]-gt_end[0])**2 + (pred_pos[1]-gt_end[1])**2))
            smallest_idx = np.argmin(distances)
            matched_end = list(predicted_graph.nodes)[smallest_idx]

            # calculate shortest distances and the apls value
            if end in gt_graph.neighbors(start):
                gt_shortest_path = nx.shortest_path(gt_graph, start, end)
                gt_min_distance = 0.
                for i in range(len(gt_shortest_path)-1):
                    gt_min_distance += gt_graph.get_edge_data(gt_shortest_path[i], 
                            gt_shortest_path[i+1], 0)['weight']
            else:
                gt_min_distance = 1
            
            if matched_end in predicted_graph.neighbors(matched_start):
                pred_shortest_path = nx.shortest_path(predicted_graph, matched_start, matched_end)
                pred_min_distance = 0.
                for i in range(len(pred_shortest_path)-1):
                    pred_min_distance += predicted_graph.get_edge_data(pred_shortest_path[i], 
                            pred_shortest_path[i+1], 0)['weight']
            else:
                pred_min_distance = 1

            C = np.abs(gt_min_distance - pred_min_distance) / gt_min_distance
            C = min(1, C)
            C_values.append(C)
    
    return 1 - np.mean(C_values)

def apls_new(gt: torch.Tensor, predicted: torch.Tensor, node_threshold: int = 12) -> float:
    """APLS: Average Path-Length Similarity. Calculates the differences between the shortest
    distances between pairs of nodes of the ground truth and predicted graphs.
    
    https://arxiv.org/abs/1807.01232
    """
    
    np_gt = gt.detach().cpu().numpy()
    np_predicted = predicted.detach().cpu().numpy()
    if np.max(np_gt) > 1:
        np_gt = np_gt // np.max(np_gt)
    if np.max(np_predicted) > 1:
        np_predicted = np_predicted // np.max(np_predicted)

    C_values = []
    for input, target in zip(np_predicted, np_gt):
        gt_graph = create_graph(Image(morph.skeletonize(target)))
        predicted_graph = create_graph(Image(morph.skeletonize(input)))
        _add_weights(gt_graph)
        _add_weights(predicted_graph)

        all_nodes = list(gt_graph.nodes)
        if len(all_nodes) == 0:
            continue
        if len(all_nodes) <= node_threshold:
            nodes_to_calculate = all_nodes
        else:
            # replace=False: no duplicate nodes; sorted for determinism
            idxs = np.linspace(0, len(all_nodes) - 1, node_threshold, dtype=int)
            nodes_to_calculate = [all_nodes[i] for i in idxs]

        pred_nodes = list(predicted_graph.nodes)

        all_pairs = _get_all_pairs_of_nodes(nodes_to_calculate)
        for start, end in all_pairs:
            # skip if no path exists in ground truth
            if not nx.has_path(gt_graph, start, end):
                continue

            gt_shortest_path = nx.shortest_path(gt_graph, start, end, weight='weight')
            gt_min_distance = sum(
                gt_graph.get_edge_data(gt_shortest_path[i], gt_shortest_path[i + 1], 0)['weight']
                for i in range(len(gt_shortest_path) - 1)
            )
            if gt_min_distance == 0:
                continue

            # match gt nodes to nearest nodes in predicted graph
            def nearest_pred_node(gt_pos):
                pred_positions = np.array([predicted_graph.nodes[n]['center'] for n in pred_nodes])
                dists = np.linalg.norm(pred_positions - np.array(gt_pos), axis=1)
                return pred_nodes[int(np.argmin(dists))]

            gt_start_pos = gt_graph.nodes[start]['center']
            gt_end_pos   = gt_graph.nodes[end]['center']
            matched_start = nearest_pred_node(gt_start_pos)
            matched_end   = nearest_pred_node(gt_end_pos)

            # if no path in predicted, maximum penalty
            if not nx.has_path(predicted_graph, matched_start, matched_end):
                C_values.append(1.0)
                continue

            pred_shortest_path = nx.shortest_path(predicted_graph, matched_start, matched_end, weight='weight')
            pred_min_distance = sum(
                predicted_graph.get_edge_data(pred_shortest_path[i], pred_shortest_path[i + 1], 0)['weight']
                for i in range(len(pred_shortest_path) - 1)
            )

            C = min(1.0, abs(gt_min_distance - pred_min_distance) / gt_min_distance)
            C_values.append(C)

    if not C_values:
        return 0.0
    return 1 - float(np.mean(C_values))

def tlts(inputs: torch.Tensor):
    # TODO
    return None
    
class MorphMetrics():

    def __init__(
            self, 
            threshold: float = 0.5,
            ccq_distance: int = 3
            ) -> None:
        
        self.threshold = threshold
        self.ccq_distance = ccq_distance

        # NOTE: We cannot ignore indexes when calculating morphology metrics, because we don't want
        # to break the structure of the images, so, be aware of the effects of torch.argmax()
        # on ignored indexes. e.g. if ignore_index == -100, these pixels will be treated as 
        # background

    def __call__(self, 
            scores: torch.Tensor, 
            targets: torch.Tensor,
            logits_flag: bool = False
            ) -> Tuple[float, float, float, float]:

        if logits_flag:
            scores = logits_to_preds(
                scores, task="binary", return_indices=True, threshold=self.threshold
            )

        return self._apply_morph_metrics(scores, targets)
    
    @torch.no_grad()
    def _apply_morph_metrics(
            self,
            scores: torch.Tensor, 
            targets: torch.Tensor
            ) -> Tuple[float, float, float, float]:

        preds = scores.to(torch.float32)

        # cl_dice_val = cl_dice(targets, preds)
        betti0, betti1 = betti_numbers(preds)
        apls_val = apls(targets, preds)
        apls_val_new = apls_new(targets, preds)
        correctness, completeness, quality = ccq(targets, preds, distance = self.ccq_distance)
        correctness_new, completeness_new, quality_new = ccq_new(targets, preds, distance = self.ccq_distance)
        
        out = {"betti": (betti0, betti1),
               "correctness": correctness_new,
               "completeness": completeness_new,
               "quality": quality_new,
               "apls": apls_val_new,
               }

        return out
        