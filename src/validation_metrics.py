"""
Ground-Truth-Based Validation Metrics System
Research-Grade Implementation with Two-Phase Evaluation

Computes comprehensive detection performance metrics including:
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Detection Delay Distribution
- False Alarm Rate per Hour
- Attack-wise confusion matrices
- Precision-Recall Curve
- Calibration curve
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef,
    precision_recall_curve, roc_curve, average_precision_score
)
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class AttackType(Enum):
    """Attack type enumeration"""
    SENSOR_SPOOFING = "sensor_spoofing"
    REPLAY_ATTACK = "replay_attack"
    GRADUAL_MANIPULATION = "gradual_manipulation"
    FROZEN_SENSOR = "frozen_sensor"
    DELAY_DOS = "delay_dos"


@dataclass
class DetectionMetrics:
    """Comprehensive detection metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    false_positive_rate: float
    false_negative_rate: float
    detection_latency_mean: float
    detection_latency_std: float
    missed_attack_rate: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    # Additional metrics
    balanced_accuracy: float = 0.0
    matthews_corrcoef: float = 0.0
    false_alarm_rate_per_hour: float = 0.0
    detection_delay_distribution: Dict[str, float] = field(default_factory=dict)
    precision_recall_auc: float = 0.0


@dataclass
class Phase1Metrics:
    """Phase 1: Normal-only validation metrics"""
    false_positive_rate: float
    false_alarm_rate_per_hour: float
    normal_samples_flagged: int
    total_normal_samples: int
    reconstruction_error_mean: float
    reconstruction_error_std: float
    threshold_value: float


class ValidationMetrics:
    """
    Computes ground-truth-based validation metrics with two-phase evaluation
    """
    
    def __init__(self):
        """Initialize validation metrics calculator"""
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_confidence_scores = []
        self.attack_type_labels = []
        self.detection_latencies = []
        self.timestamps = []  # For false alarm rate calculation
        self.attack_start_times = {}  # Track attack start times for delay calculation
        
        # Phase 1 (normal-only) data
        self.phase1_predictions = []
        self.phase1_ground_truth = []
        self.phase1_confidence_scores = []
        self.phase1_timestamps = []
        
        # Phase 2 (attack detection) data
        self.phase2_predictions = []
        self.phase2_ground_truth = []
        self.phase2_confidence_scores = []
        self.phase2_attack_types = []
        self.phase2_timestamps = []
        self.phase2_detection_delays = []
        
    def add_detection(self, predicted: bool, ground_truth: bool,
                     confidence_score: float, attack_type: Optional[str] = None,
                     detection_latency: Optional[float] = None,
                     timestamp: Optional[float] = None,
                     attack_start_time: Optional[float] = None,
                     phase: int = 2):
        """
        Add detection result for metrics computation
        
        Args:
            predicted: System prediction (True = anomaly)
            ground_truth: Actual ground truth (True = attack)
            confidence_score: Detection confidence [0, 1]
            attack_type: Type of attack (if applicable)
            detection_latency: Detection delay in seconds
            timestamp: Sample timestamp
            attack_start_time: When attack actually started
            phase: 1 for normal-only validation, 2 for attack detection
        """
        self.all_predictions.append(predicted)
        self.all_ground_truth.append(ground_truth)
        self.all_confidence_scores.append(confidence_score)
        self.attack_type_labels.append(attack_type)
        if timestamp is not None:
            self.timestamps.append(timestamp)
        
        if detection_latency is not None:
            self.detection_latencies.append(detection_latency)
        
        if attack_start_time is not None and attack_type is not None:
            if attack_type not in self.attack_start_times:
                self.attack_start_times[attack_type] = attack_start_time
        
        # Track by phase
        if phase == 1:
            self.phase1_predictions.append(predicted)
            self.phase1_ground_truth.append(ground_truth)
            self.phase1_confidence_scores.append(confidence_score)
            if timestamp is not None:
                self.phase1_timestamps.append(timestamp)
        else:
            self.phase2_predictions.append(predicted)
            self.phase2_ground_truth.append(ground_truth)
            self.phase2_confidence_scores.append(confidence_score)
            self.phase2_attack_types.append(attack_type)
            if timestamp is not None:
                self.phase2_timestamps.append(timestamp)
            if detection_latency is not None:
                self.phase2_detection_delays.append(detection_latency)
    
    def compute_phase1_metrics(self, threshold: float) -> Phase1Metrics:
        """
        Compute Phase 1 metrics (normal-only validation)
        
        Args:
            threshold: Threshold used for detection
        
        Returns:
            Phase1Metrics object
        """
        if len(self.phase1_predictions) == 0:
            raise ValueError("No Phase 1 data available")
        
        y_pred = np.array(self.phase1_predictions)
        y_true = np.array(self.phase1_ground_truth)
        y_scores = np.array(self.phase1_confidence_scores)
        
        # False positive rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False alarm rate per hour
        false_alarm_rate_per_hour = 0.0
        if len(self.phase1_timestamps) > 1:
            time_span_hours = (max(self.phase1_timestamps) - min(self.phase1_timestamps)) / 3600.0
            if time_span_hours > 0:
                false_alarm_rate_per_hour = fp / time_span_hours
        
        # Reconstruction error statistics (using confidence scores as proxy)
        reconstruction_error_mean = np.mean(y_scores)
        reconstruction_error_std = np.std(y_scores)
        
        return Phase1Metrics(
            false_positive_rate=false_positive_rate,
            false_alarm_rate_per_hour=false_alarm_rate_per_hour,
            normal_samples_flagged=int(fp),
            total_normal_samples=int(fp + tn),
            reconstruction_error_mean=reconstruction_error_mean,
            reconstruction_error_std=reconstruction_error_std,
            threshold_value=threshold
        )
    
    def compute_metrics(self) -> DetectionMetrics:
        """
        Compute comprehensive detection metrics (Phase 2)
        
        Returns:
            DetectionMetrics object
        """
        if len(self.phase2_predictions) == 0:
            # Fallback to all data if phase 2 not used
            y_pred = np.array(self.all_predictions)
            y_true = np.array(self.all_ground_truth)
            y_scores = np.array(self.all_confidence_scores)
        else:
            y_pred = np.array(self.phase2_predictions)
            y_true = np.array(self.phase2_ground_truth)
            y_scores = np.array(self.phase2_confidence_scores)
        
        if len(y_pred) == 0:
            raise ValueError("No predictions available for metrics computation")
        
        # Check class distribution
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            raise ValueError(f"Only one class present in ground truth: {unique_classes}. Cannot compute ROC-AUC.")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # ROC-AUC (only if both classes present)
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            roc_auc = 0.0
        
        # Precision-Recall AUC
        try:
            pr_auc = average_precision_score(y_true, y_scores)
        except ValueError:
            pr_auc = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
        
        # Derived metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        missed_attack_rate = false_negative_rate
        
        # Detection latency
        if len(self.phase2_detection_delays) > 0:
            detection_latency_mean = np.mean(self.phase2_detection_delays)
            detection_latency_std = np.std(self.phase2_detection_delays)
        elif len(self.detection_latencies) > 0:
            detection_latency_mean = np.mean(self.detection_latencies)
            detection_latency_std = np.std(self.detection_latencies)
        else:
            detection_latency_mean = 0.0
            detection_latency_std = 0.0
        
        # Detection delay distribution
        if len(self.phase2_detection_delays) > 0:
            delays = np.array(self.phase2_detection_delays)
            detection_delay_distribution = {
                'mean': float(np.mean(delays)),
                'std': float(np.std(delays)),
                'min': float(np.min(delays)),
                'max': float(np.max(delays)),
                'median': float(np.median(delays)),
                'q25': float(np.percentile(delays, 25)),
                'q75': float(np.percentile(delays, 75))
            }
        else:
            detection_delay_distribution = {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'median': 0.0, 'q25': 0.0, 'q75': 0.0
            }
        
        # False alarm rate per hour (from phase 2 normal samples)
        false_alarm_rate_per_hour = 0.0
        if len(self.phase2_timestamps) > 1:
            y_true = y_true[:len(self.phase2_timestamps)]   # 🔥 FIX
            normal_mask = y_true == False

            if np.sum(normal_mask) > 0:
                normal_timestamps = np.array(self.phase2_timestamps)[normal_mask]
                if len(normal_timestamps) > 1:
                    time_span_hours = (max(normal_timestamps) - min(normal_timestamps)) / 3600.0
                    if time_span_hours > 0:
                        false_alarm_rate_per_hour = fp / time_span_hours

        
        return DetectionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            detection_latency_mean=detection_latency_mean,
            detection_latency_std=detection_latency_std,
            missed_attack_rate=missed_attack_rate,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            balanced_accuracy=balanced_acc,
            matthews_corrcoef=mcc,
            false_alarm_rate_per_hour=false_alarm_rate_per_hour,
            detection_delay_distribution=detection_delay_distribution,
            precision_recall_auc=pr_auc
        )
    
    def compute_per_attack_metrics(self) -> Dict[str, DetectionMetrics]:
        """
        Compute metrics per attack type with confusion matrices
        
        Returns:
            Dictionary mapping attack type to metrics
        """
        if len(self.phase2_predictions) == 0:
            # Fallback to all data
            predictions = self.all_predictions
            ground_truth = self.all_ground_truth
            confidence_scores = self.all_confidence_scores
            attack_types = self.attack_type_labels
            timestamps = self.timestamps
        else:
            predictions = self.phase2_predictions
            ground_truth = self.phase2_ground_truth
            confidence_scores = self.phase2_confidence_scores
            attack_types = self.phase2_attack_types
            timestamps = self.phase2_timestamps
        
        attack_types_set = set(attack_types)
        attack_types_set.discard(None)
        
        per_attack_metrics = {}
        
        for attack_type in attack_types_set:
            # Filter predictions for this attack type
            mask = np.array([at == attack_type for at in attack_types])
            
            if np.sum(mask) == 0:
                continue
            
            y_pred = np.array(predictions)[mask]
            y_true = np.array(ground_truth)[mask]
            y_scores = np.array(confidence_scores)[mask]
            
            # Check if both classes present
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                # Only attacks, no normal samples for this attack type
                continue
            
            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                roc_auc = 0.0
            
            try:
                pr_auc = average_precision_score(y_true, y_scores)
            except ValueError:
                pr_auc = 0.0
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
            
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Detection latency for this attack type
            attack_delays = [self.phase2_detection_delays[i] 
                           for i in range(len(self.phase2_detection_delays))
                           if i < len(mask) and mask[i]]
            
            latency_mean = np.mean(attack_delays) if len(attack_delays) > 0 else 0.0
            latency_std = np.std(attack_delays) if len(attack_delays) > 0 else 0.0
            
            # Detection delay distribution
            if len(attack_delays) > 0:
                delays = np.array(attack_delays)
                delay_dist = {
                    'mean': float(np.mean(delays)),
                    'std': float(np.std(delays)),
                    'min': float(np.min(delays)),
                    'max': float(np.max(delays)),
                    'median': float(np.median(delays))
                }
            else:
                delay_dist = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
            
            per_attack_metrics[attack_type] = DetectionMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                detection_latency_mean=latency_mean,
                detection_latency_std=latency_std,
                missed_attack_rate=false_negative_rate,
                true_positives=int(tp),
                true_negatives=int(tn),
                false_positives=int(fp),
                false_negatives=int(fn),
                balanced_accuracy=balanced_acc,
                matthews_corrcoef=mcc,
                false_alarm_rate_per_hour=0.0,  # Per-attack doesn't need hourly rate
                detection_delay_distribution=delay_dist,
                precision_recall_auc=pr_auc
            )
        
        return per_attack_metrics
    
    def get_attack_wise_confusion_matrices(self) -> Dict[str, Dict]:
        """
        Get confusion matrices for each attack type
        
        Returns:
            Dictionary mapping attack type to confusion matrix dict
        """
        if len(self.phase2_predictions) == 0:
            predictions = self.all_predictions
            ground_truth = self.all_ground_truth
            attack_types = self.attack_type_labels
        else:
            predictions = self.phase2_predictions
            ground_truth = self.phase2_ground_truth
            attack_types = self.phase2_attack_types
        
        attack_types_set = set(attack_types)
        attack_types_set.discard(None)
        
        confusion_matrices = {}
        
        for attack_type in attack_types_set:
            mask = np.array([at == attack_type for at in attack_types])
            if np.sum(mask) == 0:
                continue
            
            y_pred = np.array(predictions)[mask]
            y_true = np.array(ground_truth)[mask]
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
            
            confusion_matrices[attack_type] = {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        
        return confusion_matrices
    
    def compute_precision_recall_curve(self) -> Dict:
        """
        Compute Precision-Recall curve data
        
        Returns:
            Dictionary with precision, recall, thresholds, and AUC
        """
        if len(self.phase2_predictions) == 0:
            y_true = np.array(self.all_ground_truth)
            y_scores = np.array(self.all_confidence_scores)
        else:
            y_true = np.array(self.phase2_ground_truth)
            y_scores = np.array(self.phase2_confidence_scores)
        
        if len(y_true) == 0:
            return {'precision': [], 'recall': [], 'thresholds': [], 'auc': 0.0}
        
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            auc = average_precision_score(y_true, y_scores)
            
            return {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(auc)
            }
        except ValueError:
            return {'precision': [], 'recall': [], 'thresholds': [], 'auc': 0.0}
    
    def compute_calibration_curve(self, n_bins: int = 10) -> Dict:
        """
        Compute calibration curve (confidence vs reality)
        
        Args:
            n_bins: Number of bins for calibration
        
        Returns:
            Dictionary with calibration data
        """
        if len(self.phase2_predictions) == 0:
            y_true = np.array(self.all_ground_truth)
            y_scores = np.array(self.all_confidence_scores)
        else:
            y_true = np.array(self.phase2_ground_truth)
            y_scores = np.array(self.phase2_confidence_scores)
        
        if len(y_true) == 0:
            return {'fraction_of_positives': [], 'mean_predicted_value': [], 'bins': []}
        
        # Bin the predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        fraction_of_positives = []
        mean_predicted_value = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            if np.sum(in_bin) > 0:
                fraction_of_positives.append(np.mean(y_true[in_bin]))
                mean_predicted_value.append(np.mean(y_scores[in_bin]))
            else:
                fraction_of_positives.append(0.0)
                mean_predicted_value.append((bin_lower + bin_upper) / 2.0)
        
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'bins': bin_boundaries.tolist()
        }
    
    def generate_metrics_table(self) -> pd.DataFrame:
        """
        Generate metrics summary table
        
        Returns:
            DataFrame with metrics
        """
        overall_metrics = self.compute_metrics()
        per_attack = self.compute_per_attack_metrics()
        
        # Create table
        data = {
            'Metric': [
                'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score',
                'ROC-AUC', 'PR-AUC', 'MCC', 'False Positive Rate', 'False Negative Rate',
                'Missed Attack Rate', 'False Alarm Rate (per hour)',
                'Detection Latency (mean)', 'Detection Latency (std)'
            ],
            'Overall': [
                f"{overall_metrics.accuracy:.4f}",
                f"{overall_metrics.balanced_accuracy:.4f}",
                f"{overall_metrics.precision:.4f}",
                f"{overall_metrics.recall:.4f}",
                f"{overall_metrics.f1_score:.4f}",
                f"{overall_metrics.roc_auc:.4f}",
                f"{overall_metrics.precision_recall_auc:.4f}",
                f"{overall_metrics.matthews_corrcoef:.4f}",
                f"{overall_metrics.false_positive_rate:.4f}",
                f"{overall_metrics.false_negative_rate:.4f}",
                f"{overall_metrics.missed_attack_rate:.4f}",
                f"{overall_metrics.false_alarm_rate_per_hour:.4f}",
                f"{overall_metrics.detection_latency_mean:.2f}s",
                f"{overall_metrics.detection_latency_std:.2f}s"
            ]
        }
        
        # Add per-attack columns
        for attack_type, metrics in per_attack.items():
            data[attack_type] = [
                f"{metrics.accuracy:.4f}",
                f"{metrics.balanced_accuracy:.4f}",
                f"{metrics.precision:.4f}",
                f"{metrics.recall:.4f}",
                f"{metrics.f1_score:.4f}",
                f"{metrics.roc_auc:.4f}",
                f"{metrics.precision_recall_auc:.4f}",
                f"{metrics.matthews_corrcoef:.4f}",
                f"{metrics.false_positive_rate:.4f}",
                f"{metrics.false_negative_rate:.4f}",
                f"{metrics.missed_attack_rate:.4f}",
                "N/A",  # Per-attack doesn't have hourly rate
                f"{metrics.detection_latency_mean:.2f}s",
                f"{metrics.detection_latency_std:.2f}s"
            ]
        
        return pd.DataFrame(data)
    
    def generate_confusion_matrix_table(self) -> pd.DataFrame:
        """
        Generate confusion matrix summary
        
        Returns:
            DataFrame with confusion matrix
        """
        overall_metrics = self.compute_metrics()
        
        data = {
            '': ['Predicted Normal', 'Predicted Attack'],
            'Actual Normal': [
                overall_metrics.true_negatives,
                overall_metrics.false_positives
            ],
            'Actual Attack': [
                overall_metrics.false_negatives,
                overall_metrics.true_positives
            ]
        }
        
        return pd.DataFrame(data)
    
    def reset(self):
        """Reset all metrics"""
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_confidence_scores = []
        self.attack_type_labels = []
        self.detection_latencies = []
        self.timestamps = []
        self.attack_start_times = {}
        
        self.phase1_predictions = []
        self.phase1_ground_truth = []
        self.phase1_confidence_scores = []
        self.phase1_timestamps = []
        
        self.phase2_predictions = []
        self.phase2_ground_truth = []
        self.phase2_confidence_scores = []
        self.phase2_attack_types = []
        self.phase2_timestamps = []
        self.phase2_detection_delays = []


if __name__ == "__main__":
    # Example usage
    validator = ValidationMetrics()
    
    # Simulate detections
    for i in range(100):
        ground_truth = i % 10 == 0  # 10% attack rate
        predicted = (i % 10 == 0) or (i % 15 == 0)  # Some false positives
        confidence = 0.8 if predicted else 0.2
        validator.add_detection(predicted, ground_truth, confidence)
    
    # Compute metrics
    metrics = validator.compute_metrics()
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1-Score: {metrics.f1_score:.4f}")
    print(f"Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
    print(f"MCC: {metrics.matthews_corrcoef:.4f}")
    
    # Generate table
    table = validator.generate_metrics_table()
    print("\nMetrics Table:")
    print(table)
