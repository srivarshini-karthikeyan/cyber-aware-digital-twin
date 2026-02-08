"""
Adaptive Threshold Learning System

Automatically adjusts anomaly thresholds based on evolving data patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
import yaml
from dataclasses import dataclass


@dataclass
class ThresholdState:
    """Current threshold state"""
    value: float
    adaptation_rate: float
    confidence: float
    last_update: float


class AdaptiveThreshold:
    """
    Adaptive threshold that learns from data evolution
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 initial_threshold: float = 0.15,
                 adaptation_rate: float = 0.1):
        """Initialize adaptive threshold"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        
        # Current threshold state
        self.current_threshold = initial_threshold
        self.threshold_history: List[float] = [initial_threshold]
        
        # Error history for adaptation
        self.error_history: deque = deque(maxlen=1000)
        self.reconstruction_errors: deque = deque(maxlen=1000)
        
        # Drift detection
        self.drift_detected = False
        self.drift_magnitude = 0.0
        
        # Adaptation parameters
        self.min_threshold = 0.05
        self.max_threshold = 0.5
        self.adaptation_window = 100  # Samples
        
    def update(self, reconstruction_error: float, is_anomaly: bool,
              timestamp: float):
        """
        Update threshold based on new error observation
        
        Args:
            reconstruction_error: Current reconstruction error
            is_anomaly: Whether this was classified as anomaly
            timestamp: Current timestamp
        """
        self.reconstruction_errors.append(reconstruction_error)
        self.error_history.append({
            'error': reconstruction_error,
            'is_anomaly': is_anomaly,
            'timestamp': timestamp
        })
        
        # Check for drift
        self._detect_drift()
        
        # Adapt threshold
        self._adapt_threshold()
        
        # Record threshold
        self.threshold_history.append(self.current_threshold)
    
    def _detect_drift(self):
        """Detect behavioral drift in data"""
        if len(self.reconstruction_errors) < self.adaptation_window:
            return
        
        # Compare recent errors to historical baseline
        recent_errors = list(self.reconstruction_errors)[-self.adaptation_window:]
        baseline_errors = list(self.reconstruction_errors)[:self.adaptation_window]
        
        recent_mean = np.mean(recent_errors)
        baseline_mean = np.mean(baseline_errors)
        
        # Drift detection: significant shift in mean
        drift_magnitude = abs(recent_mean - baseline_mean) / (baseline_mean + 1e-10)
        
        if drift_magnitude > 0.2:  # 20% shift
            self.drift_detected = True
            self.drift_magnitude = drift_magnitude
        else:
            self.drift_detected = False
            self.drift_magnitude = 0.0
    
    def _adapt_threshold(self):
        """Adapt threshold based on error distribution"""
        if len(self.reconstruction_errors) < 50:
            return  # Need enough data
        
        # Compute percentile-based threshold
        errors = np.array(list(self.reconstruction_errors))
        
        # Use 95th percentile as adaptive threshold
        percentile_threshold = np.percentile(errors, 95)
        
        # Smooth adaptation
        self.current_threshold = (
            (1 - self.adaptation_rate) * self.current_threshold +
            self.adaptation_rate * percentile_threshold
        )
        
        # Clamp to bounds
        self.current_threshold = np.clip(
            self.current_threshold,
            self.min_threshold,
            self.max_threshold
        )
    
    def get_threshold(self) -> float:
        """Get current adaptive threshold"""
        return self.current_threshold
    
    def get_threshold_state(self) -> ThresholdState:
        """Get current threshold state"""
        return ThresholdState(
            value=self.current_threshold,
            adaptation_rate=self.adaptation_rate,
            confidence=1.0 - min(self.drift_magnitude, 1.0),
            last_update=len(self.threshold_history) - 1
        )
    
    def reset(self):
        """Reset adaptive threshold"""
        self.current_threshold = self.initial_threshold
        self.threshold_history = [self.initial_threshold]
        self.error_history.clear()
        self.reconstruction_errors.clear()
        self.drift_detected = False
        self.drift_magnitude = 0.0


class BehavioralDriftDetector:
    """
    Detects long-term behavioral drift in system
    """
    
    def __init__(self, window_size: int = 1000):
        """Initialize drift detector"""
        self.window_size = window_size
        self.behavior_history: deque = deque(maxlen=window_size)
        self.drift_events: List[Dict] = []
        
    def add_observation(self, feature_vector: np.ndarray, timestamp: float):
        """
        Add new observation for drift detection
        
        Args:
            feature_vector: Feature vector at this time
            timestamp: Observation timestamp
        """
        self.behavior_history.append({
            'features': feature_vector,
            'timestamp': timestamp
        })
        
        # Check for drift
        if len(self.behavior_history) >= self.window_size:
            drift_detected = self._check_drift()
            if drift_detected:
                self.drift_events.append({
                    'timestamp': timestamp,
                    'magnitude': self._compute_drift_magnitude()
                })
    
    def _check_drift(self) -> bool:
        """Check if drift is detected"""
        if len(self.behavior_history) < self.window_size:
            return False
        
        # Compare first half to second half
        mid_point = len(self.behavior_history) // 2
        first_half = [obs['features'] for obs in 
                     list(self.behavior_history)[:mid_point]]
        second_half = [obs['features'] for obs in 
                     list(self.behavior_history)[mid_point:]]
        
        first_mean = np.mean(first_half, axis=0)
        second_mean = np.mean(second_half, axis=0)
        
        # Compute drift magnitude
        drift = np.linalg.norm(second_mean - first_mean)
        threshold = np.linalg.norm(first_mean) * 0.1  # 10% change
        
        return drift > threshold
    
    def _compute_drift_magnitude(self) -> float:
        """Compute magnitude of detected drift"""
        if len(self.behavior_history) < self.window_size:
            return 0.0
        
        mid_point = len(self.behavior_history) // 2
        first_half = [obs['features'] for obs in 
                     list(self.behavior_history)[:mid_point]]
        second_half = [obs['features'] for obs in 
                     list(self.behavior_history)[mid_point:]]
        
        first_mean = np.mean(first_half, axis=0)
        second_mean = np.mean(second_half, axis=0)
        
        drift = np.linalg.norm(second_mean - first_mean)
        baseline = np.linalg.norm(first_mean)
        
        return drift / (baseline + 1e-10)
    
    def get_drift_summary(self) -> Dict:
        """Get summary of drift events"""
        return {
            'total_drift_events': len(self.drift_events),
            'latest_drift': self.drift_events[-1] if self.drift_events else None,
            'average_drift_magnitude': np.mean([e['magnitude'] for e in self.drift_events]) 
                                      if self.drift_events else 0.0
        }


if __name__ == "__main__":
    # Example usage
    adaptive = AdaptiveThreshold(initial_threshold=0.15)
    
    # Simulate error updates
    for i in range(200):
        error = 0.1 + np.random.randn() * 0.05
        if i > 100:
            error += 0.1  # Simulate drift
        is_anomaly = error > adaptive.get_threshold()
        adaptive.update(error, is_anomaly, float(i))
        
        if i % 20 == 0:
            print(f"t={i}: Threshold={adaptive.get_threshold():.4f}, "
                  f"Drift={adaptive.drift_detected}")
