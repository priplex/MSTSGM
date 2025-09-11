import os
import json
import numpy as np
from typing import List, Dict, Tuple


def calculate_iou(box1: Tuple[float], box2: Tuple[float]) -> float:
    """
    Calculate IoU between two bounding boxes (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_ap(
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
) -> float:
    """
    Calculate AP for a single class
    """
    # Sort predictions by confidence (descending)
    predictions_sorted = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

    # Initialize variables
    tp = np.zeros(len(predictions_sorted))
    fp = np.zeros(len(predictions_sorted))
    gt_used = [False] * len(ground_truths)

    # Iterate over sorted predictions
    for i, pred in enumerate(predictions_sorted):
        pred_box = pred['bbox']
        max_iou = 0.0
        max_idx = -1

        # Find best matching ground truth
        for j, gt in enumerate(ground_truths):
            if not gt_used[j]:
                iou = calculate_iou(pred_box, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j

        # Determine TP/FP
        if max_iou >= iou_threshold and max_idx != -1:
            tp[i] = 1
            gt_used[max_idx] = True
        else:
            fp[i] = 1

    # Calculate cumulative precision and recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)  # Avoid division by zero
    recall = cum_tp / (len(ground_truths) + 1e-10)

    # Add 0 at the beginning for numerical stability
    precision = np.concatenate([[0], precision])
    recall = np.concatenate([[0], recall])

    # Calculate AP using interpolation
    ap = 0.0
    for i in range(1, len(precision)):
        ap += precision[i] * (recall[i] - recall[i - 1])

    return ap


def calculate_map(
        all_predictions: Dict[int, List[Dict]],
        all_ground_truths: Dict[int, List[Dict]],
        iou_threshold: float = 0.5
) -> float:
    """
    Calculate mean AP across all classes
    """
    # Get all unique classes
    classes = set(all_predictions.keys()).union(set(all_ground_truths.keys()))
    aps = []

    for cls in classes:
        preds = all_predictions.get(cls, [])
        gts = all_ground_truths.get(cls, [])

        if len(gts) == 0:
            continue  # Skip classes with no ground truths

        ap = calculate_ap(preds, gts, iou_threshold)
        aps.append(ap)
        print(f"Class {cls} AP@0.5: {ap:.4f}")

    return np.mean(aps) if aps else 0.0


def load_yolov8_results(pred_dir: str) -> Dict[int, List[Dict]]:
    """
    Load YOLOv8 predictions from txt files (format: class_id conf x1 y1 x2 y2)
    """
    predictions = {}
    for filename in os.listdir(pred_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(pred_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue

                    cls_id = int(parts[0])
                    conf = float(parts[1])
                    bbox = tuple(map(float, parts[2:6]))  # (x1, y1, x2, y2)

                    if cls_id not in predictions:
                        predictions[cls_id] = []
                    predictions[cls_id].append({
                        'bbox': bbox,
                        'confidence': conf
                    })
    return predictions


def load_ground_truths(gt_dir: str) -> Dict[int, List[Dict]]:
    """
    Load ground truth annotations from txt files (format: class_id x1 y1 x2 y2)
    """
    ground_truths = {}
    for filename in os.listdir(gt_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(gt_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls_id = int(parts[0])
                    bbox = tuple(map(float, parts[1:5]))  # (x1, y1, x2, y2)

                    if cls_id not in ground_truths:
                        ground_truths[cls_id] = []
                    ground_truths[cls_id].append({'bbox': bbox})
    return ground_truths


# Example usage
if __name__ == "__main__":
    # Configuration
    PREDICTIONS_DIR = "yolov8_predictions"  # Directory with YOLOv8 .txt results
    GROUND_TRUTHS_DIR = "ground_truths"  # Directory with ground truth .txt files
    IOU_THRESHOLD = 0.5

    # Load data
    print("Loading predictions...")
    predictions = load_yolov8_results(PREDICTIONS_DIR)

    print("Loading ground truths...")
    ground_truths = load_ground_truths(GROUND_TRUTHS_DIR)

    # Calculate mAP@0.5
    map_score = calculate_map(predictions, ground_truths, IOU_THRESHOLD)
    print(f"\nAverage mAP@0.5: {map_score:.4f}")
