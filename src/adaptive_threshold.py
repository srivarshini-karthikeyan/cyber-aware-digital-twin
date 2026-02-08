"""
Adaptive Thresholding System
Elite Enhancement for Production IDS

Dynamically adjusts detection threshold based on:
- Recent false positive/negative rates
- System operational context
- Time-based patterns
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, time
import json


@dataclass
class ThresholdMetrics:
    """Metrics for threshold adaptation"""
    false_positive_rate: float
    false_negative_rate: float
    true_positive_rate: float
    true_negative_rate: float
    sample_count: int
    timestamp: float


class AdaptiveThreshold:
    """
    Adaptive threshold system that adjusts based on performance
    """
    
    def __init__(self,
                 base_threshold: float,
                 target_fpr: float = 0.05,  # 5% target FPR
                 target_fnr: float = 0.15,  # 15% target FNR
                 adaptation_rate: float = 0.1,
                 min_threshold: Optional[float] = None,
                 max_threshold: Optional[float] = None,
                 history_window: int = 100):
        """
        Initialize adaptive threshold
        
        Args:
            base_threshold: Initial threshold (from Phase 1 calibration)
            target_fpr: Target false positive rate
            target_fnr: Target false negative rate
            adaptation_rate: How quickly to adapt (0.1 = 10% adjustment per cycle)
            min_threshold: Minimum allowed threshold (80% of base)
            max_threshold: Maximum allowed threshold (120% of base)
            history_window: Number of recent samples to consider
        """
        # Ensure numeric types
        base_threshold = float(base_threshold)

        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.target_fpr = float(target_fpr)
        self.target_fnr = float(target_fnr)
        self.adaptation_rate = float(adaptation_rate)

        # Clamp bounds
        self.min_threshold = float(min_threshold) if min_threshold is not None else (base_threshold * 0.8)
        self.max_threshold = float(max_threshold) if max_threshold is not None else (base_threshold * 1.2)

        
        # Performance history
        self.history_window = history_window
        self.recent_predictions: List[bool] = []
        self.recent_ground_truth: List[bool] = []
        self.metrics_history: List[ThresholdMetrics] = []
        
        # Context tracking
        self.operational_context = "normal"  # normal, maintenance, startup, shutdown
        self.drift_detected = False

    
    def update_performance(self, predicted: bool, ground_truth: bool):
        """
        Update performance metrics with new prediction
        
        Args:
            predicted: System prediction (True = anomaly)
            ground_truth: Actual ground truth (True = attack)
        """
        self.recent_predictions.append(predicted)
        self.recent_ground_truth.append(ground_truth)
        
        # Keep only recent history
        if len(self.recent_predictions) > self.history_window:
            self.recent_predictions = self.recent_predictions[-self.history_window:]
            self.recent_ground_truth = self.recent_ground_truth[-self.history_window:]
    
    def compute_recent_metrics(self) -> ThresholdMetrics:
        """Compute metrics from recent predictions"""
        if len(self.recent_predictions) == 0:
            return ThresholdMetrics(
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                true_positive_rate=0.0,
                true_negative_rate=0.0,
                sample_count=0,
                timestamp=datetime.now().timestamp()
            )
        
        y_pred = np.array(self.recent_predictions)
        y_true = np.array(self.recent_ground_truth)
        
        # Compute confusion matrix
        tp = np.sum((y_pred == True) & (y_true == True))
        tn = np.sum((y_pred == False) & (y_true == False))
        fp = np.sum((y_pred == True) & (y_true == False))
        fn = np.sum((y_pred == False) & (y_true == True))
        
        # Compute rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = ThresholdMetrics(
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            true_positive_rate=tpr,
            true_negative_rate=tnr,
            sample_count=len(y_pred),
            timestamp=datetime.now().timestamp()
        )
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def adapt_threshold(self, context: Optional[str] = None) -> float:
        if context:
            self.operational_context = context

        # Get recent metrics
        metrics = self.compute_recent_metrics()

        if metrics.sample_count < 20:
            self.drift_detected = False
            return self.current_threshold

        # -----------------------------
        # ✅ DRIFT DETECTION (ADD HERE)
        # -----------------------------
        self.drift_detected = (
            abs(metrics.false_positive_rate - self.target_fpr) > 0.1 or
            abs(metrics.false_negative_rate - self.target_fnr) > 0.1
        )

        # Compute adjustments
        fpr_error = metrics.false_positive_rate - self.target_fpr
        fnr_error = metrics.false_negative_rate - self.target_fnr

        if fpr_error > 0.01:
            self.current_threshold *= (1 + self.adaptation_rate * fpr_error)

        if fnr_error > 0.01:
            self.current_threshold *= (1 - self.adaptation_rate * fnr_error)

        if self.operational_context == "maintenance":
            self.current_threshold *= 1.1
        elif self.operational_context == "startup":
            self.current_threshold *= 1.05

        self.current_threshold = np.clip(
            self.current_threshold,
            self.min_threshold,
            self.max_threshold
        )

        return self.current_threshold

    
    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.current_threshold
    
    def reset_to_base(self):
        """Reset threshold to base value"""
        self.current_threshold = self.base_threshold
        self.recent_predictions = []
        self.recent_ground_truth = []
        self.metrics_history = []
    
    def get_adaptation_status(self) -> Dict:
        """Get current adaptation status"""
        metrics = self.compute_recent_metrics()
        
        return {
            'current_threshold': self.current_threshold,
            'base_threshold': self.base_threshold,
            'threshold_change_percent': ((self.current_threshold - self.base_threshold) / self.base_threshold) * 100,
            'recent_fpr': metrics.false_positive_rate,
            'recent_fnr': metrics.false_negative_rate,
            'target_fpr': self.target_fpr,
            'target_fnr': self.target_fnr,
            'sample_count': metrics.sample_count,
            'operational_context': self.operational_context,
            'adaptation_active': metrics.sample_count >= 20,
            'drift_detected': self.drift_detected
        }

    def get_threshold_state(self) -> Dict:
        """
        Get current threshold state (for dashboard / demo)
        """
        return {
            "current_threshold": self.current_threshold,
            "base_threshold": self.base_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "target_fpr": self.target_fpr,
            "target_fnr": self.target_fnr,
            "operational_context": self.operational_context,
            "samples_seen": len(self.recent_predictions),
            "adaptation_active": len(self.recent_predictions) >= 20
        }


if __name__ == "__main__":
    # Test adaptive thresholding
    base_threshold = 0.15
    adaptive = AdaptiveThreshold(
        base_threshold=base_threshold,
        target_fpr=0.05,
        target_fnr=0.15
    )
    
    # Simulate high false positive rate
    print("Simulating high FPR scenario...")
    for i in range(50):
        # Many false positives (predicted=True, actual=False)
        adaptive.update_performance(True, False)
    
    adaptive.adapt_threshold()
    status = adaptive.get_adaptation_status()
    print(f"After high FPR: threshold={status['current_threshold']:.4f}, "
          f"FPR={status['recent_fpr']:.4f}, change={status['threshold_change_percent']:.1f}%")
    
    # Reset and simulate high false negative rate
    adaptive.reset_to_base()
    print("\nSimulating high FNR scenario...")
    for i in range(50):
        # Many false negatives (predicted=False, actual=True)
        adaptive.update_performance(False, True)
    
    adaptive.adapt_threshold()
    status = adaptive.get_adaptation_status()
    print(f"After high FNR: threshold={status['current_threshold']:.4f}, "
          f"FNR={status['recent_fnr']:.4f}, change={status['threshold_change_percent']:.1f}%")
