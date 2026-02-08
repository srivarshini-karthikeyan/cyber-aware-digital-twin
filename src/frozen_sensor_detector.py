"""
Frozen Sensor Detection Module
Elite Enhancement for Production IDS

Detects sensors stuck at constant values using:
1. Rate-of-change monitoring (rule-based)
2. Variance analysis (statistical)
3. Digital twin divergence (physics-based)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FrozenSensorSeverity(Enum):
    """Frozen sensor severity levels"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"  # Low variance but within noise
    FROZEN = "frozen"  # Confirmed frozen
    CRITICAL = "critical"  # Frozen + safety impact


@dataclass
class FrozenSensorResult:
    """Frozen sensor detection result"""
    is_frozen: bool
    severity: FrozenSensorSeverity
    confidence: float  # [0, 1]
    frozen_duration: float  # Seconds sensor has been frozen
    rate_of_change: float  # Current rate of change
    variance: float  # Variance of recent values
    expected_change: float  # Expected change based on system state
    actual_change: float  # Actual observed change
    divergence_score: float  # [0, 1] - how much actual differs from expected


class FrozenSensorDetector:
    """
    Elite frozen sensor detection combining multiple techniques
    """
    
    def __init__(self, 
                 noise_threshold: float = 0.01,
                 frozen_window: int = 10,
                 variance_threshold: float = 0.001):
        """
        Initialize frozen sensor detector
        
        Args:
            noise_threshold: Maximum rate of change considered "noise" (1% of range)
            frozen_window: Number of samples to check for frozen state
            variance_threshold: Minimum variance to consider sensor "active"
        """
        self.noise_threshold = noise_threshold
        self.frozen_window = frozen_window
        self.variance_threshold = variance_threshold
        
        # History tracking
        self.sensor_history: Dict[str, list] = {}
        self.frozen_start_times: Dict[str, float] = {}
    
    def detect_frozen_sensor(self,
                            sensor_id: str,
                            sensor_values: np.ndarray,
                            timestamp: float,
                            system_state: Optional[Dict] = None) -> FrozenSensorResult:
        """
        Detect if sensor is frozen using multiple techniques
        
        Args:
            sensor_id: Unique sensor identifier
            sensor_values: Recent sensor readings (last N samples)
            timestamp: Current timestamp
            system_state: Optional system state (valve, pump, etc.) for expected change
        
        Returns:
            FrozenSensorResult with detection outcome
        """
        # Update history
        if sensor_id not in self.sensor_history:
            self.sensor_history[sensor_id] = []
        
        self.sensor_history[sensor_id].extend(sensor_values.tolist())
        
        # Keep only recent history
        max_history = self.frozen_window * 2
        if len(self.sensor_history[sensor_id]) > max_history:
            self.sensor_history[sensor_id] = self.sensor_history[sensor_id][-max_history:]
        
        recent_values = np.array(self.sensor_history[sensor_id][-self.frozen_window:])
        
        if len(recent_values) < self.frozen_window:
            return FrozenSensorResult(
                is_frozen=False,
                severity=FrozenSensorSeverity.NORMAL,
                confidence=0.0,
                frozen_duration=0.0,
                rate_of_change=0.0,
                variance=np.var(recent_values) if len(recent_values) > 0 else 1.0,
                expected_change=0.0,
                actual_change=0.0,
                divergence_score=0.0
            )
        
        # Technique 1: Rate-of-change analysis
        rate_of_change = np.abs(np.diff(recent_values))
        max_rate = np.max(rate_of_change) if len(rate_of_change) > 0 else 0.0
        mean_rate = np.mean(rate_of_change) if len(rate_of_change) > 0 else 0.0
        
        # Technique 2: Variance analysis
        variance = np.var(recent_values)
        std_dev = np.std(recent_values)
        
        # Technique 3: Expected vs actual change (if system state provided)
        expected_change = 0.0
        actual_change = 0.0
        divergence_score = 0.0
        
        if system_state is not None:
            # Calculate expected change based on system state
            # Example: If valve is open, level should increase
            valve_open = system_state.get('valve', 0) == 1
            pump_on = system_state.get('pump', 0) == 1
            
            # Simplified physics: valve adds, pump removes
            if valve_open and not pump_on:
                expected_change = 0.5  # Positive change expected
            elif pump_on and not valve_open:
                expected_change = -0.5  # Negative change expected
            elif valve_open and pump_on:
                expected_change = 0.0  # Balanced
            else:
                expected_change = 0.0  # No change expected
            
            # Actual change over window
            actual_change = recent_values[-1] - recent_values[0]
            
            # Divergence: how much actual differs from expected
            if abs(expected_change) > 0.01:  # If change is expected
                divergence = abs(actual_change - expected_change)
                divergence_score = min(1.0, divergence / abs(expected_change))
            else:  # No change expected, but check if there is change
                if abs(actual_change) > self.noise_threshold:
                    divergence_score = min(1.0, abs(actual_change) / 0.1)
        
        # Combine evidence
        is_frozen = False
        severity = FrozenSensorSeverity.NORMAL
        confidence = 0.0
        
        # Rule 1: Zero or near-zero rate of change
        if max_rate < self.noise_threshold and mean_rate < self.noise_threshold * 0.5:
            frozen_evidence = 0.6
        
        # Rule 2: Very low variance
        if variance < self.variance_threshold:
            frozen_evidence = 0.4
        else:
            frozen_evidence = 0.0
        
        # Rule 3: Divergence from expected (if system state available)
        if system_state is not None and divergence_score > 0.7:
            # Expected change but sensor shows none
            frozen_evidence += 0.3
        
        # Rule 4: Sustained constant value
        if len(recent_values) >= 5:
            unique_values = len(np.unique(recent_values))
            if unique_values <= 2:  # Only 1-2 unique values
                frozen_evidence += 0.2
        
        confidence = min(1.0, frozen_evidence)
        
        # Determine severity
        if confidence > 0.8:
            is_frozen = True
            if divergence_score > 0.8 and system_state is not None:
                severity = FrozenSensorSeverity.CRITICAL
            else:
                severity = FrozenSensorSeverity.FROZEN
        elif confidence > 0.5:
            severity = FrozenSensorSeverity.SUSPICIOUS
        else:
            severity = FrozenSensorSeverity.NORMAL
        
        # Track frozen duration
        frozen_duration = 0.0
        if is_frozen:
            if sensor_id not in self.frozen_start_times:
                self.frozen_start_times[sensor_id] = timestamp
            frozen_duration = timestamp - self.frozen_start_times[sensor_id]
        else:
            if sensor_id in self.frozen_start_times:
                del self.frozen_start_times[sensor_id]
        
        return FrozenSensorResult(
            is_frozen=is_frozen,
            severity=severity,
            confidence=confidence,
            frozen_duration=frozen_duration,
            rate_of_change=mean_rate,
            variance=variance,
            expected_change=expected_change,
            actual_change=actual_change,
            divergence_score=divergence_score
        )
    
    def reset_sensor_history(self, sensor_id: Optional[str] = None):
        """Reset history for a sensor or all sensors"""
        if sensor_id is None:
            self.sensor_history = {}
            self.frozen_start_times = {}
        else:
            if sensor_id in self.sensor_history:
                del self.sensor_history[sensor_id]
            if sensor_id in self.frozen_start_times:
                del self.frozen_start_times[sensor_id]


if __name__ == "__main__":
    # Test frozen sensor detection
    detector = FrozenSensorDetector()
    
    # Simulate normal sensor
    normal_values = np.array([500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0, 508.0, 509.0])
    result = detector.detect_frozen_sensor(
        "sensor_1", normal_values, 100.0,
        system_state={'valve': 1, 'pump': 0}
    )
    print(f"Normal sensor: Frozen={result.is_frozen}, Confidence={result.confidence:.2f}")
    
    # Simulate frozen sensor
    frozen_values = np.array([500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0])
    result = detector.detect_frozen_sensor(
        "sensor_2", frozen_values, 100.0,
        system_state={'valve': 1, 'pump': 0}  # Valve open but no change
    )
    print(f"Frozen sensor: Frozen={result.is_frozen}, Confidence={result.confidence:.2f}, Severity={result.severity.value}")
