import logging
import torch

# from convert_to_graph import convert_img_to_graph
from metrics_selection import Metrics

from metrics.topology_metrics import ClDice
from metrics.confusion_metrics import ConfusionMatrixMetrics, AveragePrecisionScore, MathewsCorrcoef, ROCAUCScore
from metrics.distance_metrics import DistanceMetrics
from metrics.hce_metric import HCEMetric
from metrics.morph_metrics import MorphMetrics
from metrics.graph_metrics import calculate_graph_distances, calculate_bioproperty


logger = logging.getLogger(__name__)

def run_metrics(scores: torch.Tensor, targets: torch.Tensor, metrics_ctrl: Metrics, task: str = "binary", print_log = False):
    if print_log:
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    results = run_image_metrics(scores, targets, metrics_ctrl, task)

    if metrics_ctrl.is_active("graphDistances"):
        logger.info("Converting images to graphs...")
        scores_graph = convert_img_to_graph(scores)
        targets_graph = convert_img_to_graph(targets)

        logger.info("Calculating graph distances...")
        results["graph_metrics"] = calculate_graph_distances(scores_graph, targets_graph)
        results["bio_metrics"] = calculate_bioproperty(scores_graph)

    return results

def run_image_metrics(scores: torch.Tensor, targets: torch.Tensor, metrics_ctrl: Metrics, task: str):
    metrics_dict = {}

    inputs = {
        "4d": (ensure_ndim(scores, 4), ensure_ndim(targets, 4)),
        "3d": (ensure_ndim(scores, 3), ensure_ndim(targets, 3)),
        "mixed_roc": (ensure_ndim(scores, 4), ensure_ndim(targets, 3))
    }

    metric_registry = {
        "clDice":          (ClDice,                 "4d",        {}),
        "MCC":             (MathewsCorrcoef,        "4d",        {}),
        "confusionMatrix": (ConfusionMatrixMetrics, "4d",        {}),
        "avgPrecision":    (AveragePrecisionScore,  "4d",        {"task": task}),
        "morph":           (MorphMetrics,           "4d",        {}),
        "distance":        (DistanceMetrics,        "mixed_roc", {}), 
        "HCE":             (HCEMetric,              "3d",        {}),
        "ROCAUC":          (ROCAUCScore,            "mixed_roc", {"task": task}),
    }

    for key, (MetricClass, input_type, kwargs) in metric_registry.items():
        if metrics_ctrl.is_active(key):
            logger.info(f"Calculating {key}")
            
            metric_fn = MetricClass(**kwargs)
            
            s_in, t_in = inputs[input_type]
            
            metrics_dict[key] = metric_fn(s_in, t_in)

    return metrics_dict

def ensure_ndim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_tensor = tensor
    while current_tensor.ndim < target_dim:
        current_tensor = current_tensor.unsqueeze(0)
    return current_tensor