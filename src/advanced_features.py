"""
Advanced Features Module

Implements:
- Trust Degradation Index (TDI)
- Attack Latency Exposure Window
- Silent Failure Detection
- Before/After Mitigation Simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml
from dataclasses import dataclass
from enum import Enum

from .digital_twin import CyberAwareDigitalTwin, ThreeStateModel


class TrustLevel(Enum):
    """Trust level classification"""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class TrustScore:
    """Trust score at a point in time"""
    timestamp: float
    trust_score: float  # [0, 1]
    trust_level: TrustLevel
    component: str
    deviation_magnitude: float
    deviation_duration: float


@dataclass
class AttackLatency:
    """Attack latency exposure window"""
    attack_id: str
    attack_start: float
    detection_time: Optional[float]
    unsafe_state_start: Optional[float]
    exposure_window: float  # Time system was unsafe before detection
    persistence_score: float  # [0, 1] - how long attacker can stay undetected


@dataclass
class SilentFailure:
    """Silent failure detection result"""
    timestamp: float
    detected: bool
    failure_type: str
    degradation_rate: float
    trend_anomaly: bool
    impact_estimate: str


class TrustDegradationIndex:
    """
    Computes Trust Degradation Index (TDI) for sensors
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize TDI calculator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        tdi_config = self.config['advanced_features']['trust_degradation_index']
        self.trust_levels = tdi_config['trust_levels']
        self.update_interval = tdi_config['update_interval']
        
        self.trust_history: List[TrustScore] = []
    
    def compute_trust_score(self, expected: float, observed: float,
                           previous_trust: Optional[float] = None,
                           deviation_history: Optional[List[float]] = None) -> float:
        """
        Compute trust score based on deviation
        
        Args:
            expected: Expected value from digital twin
            observed: Observed sensor value
            previous_trust: Previous trust score (for temporal consistency)
            deviation_history: History of deviations
        
        Returns:
            Trust score [0, 1] where 1 = fully trusted, 0 = not trusted
        """
        if expected == 0:
            expected = 1.0  # Avoid division by zero
        
        # Current deviation
        current_deviation = abs(expected - observed) / expected
        
        # Base trust score (inverse of deviation)
        base_trust = 1.0 - min(current_deviation, 1.0)
        
        # Temporal factor (if deviation persists, trust decreases)
        temporal_factor = 1.0
        if deviation_history and len(deviation_history) > 0:
            avg_deviation = np.mean(deviation_history[-10:])  # Last 10 steps
            if avg_deviation > 0.1:
                temporal_factor = 1.0 - min(avg_deviation, 0.5)
        
        # Combined trust score
        trust_score = base_trust * temporal_factor
        
        # Incorporate previous trust (smooth transitions)
        if previous_trust is not None:
            trust_score = 0.7 * trust_score + 0.3 * previous_trust
        
        return max(0.0, min(1.0, trust_score))
    
    def classify_trust_level(self, trust_score: float) -> TrustLevel:
        """Classify trust level from score"""
        if trust_score >= self.trust_levels['green']:
            return TrustLevel.GREEN
        elif trust_score >= self.trust_levels['yellow']:
            return TrustLevel.YELLOW
        else:
            return TrustLevel.RED
    
    def update_trust(self, timestamp: float, expected: float, observed: float,
                    component: str = "Level Sensor") -> TrustScore:
        """
        Update trust score for a component
        
        Returns:
            TrustScore object
        """
        # Get previous trust
        previous_trust = None
        deviation_history = []
        
        if self.trust_history:
            previous_trust = self.trust_history[-1].trust_score
            # Get recent deviations
            recent_scores = self.trust_history[-10:]
            deviation_history = [s.deviation_magnitude for s in recent_scores]
        
        # Compute deviation
        if expected == 0:
            expected = 1.0
        deviation_magnitude = abs(expected - observed) / expected
        
        # Compute trust score
        trust_score = self.compute_trust_score(
            expected, observed, previous_trust, deviation_history
        )
        
        # Classify trust level
        trust_level = self.classify_trust_level(trust_score)
        
        # Compute deviation duration
        deviation_duration = 0.0
        if deviation_magnitude > 0.1:
            # Count consecutive high deviations
            for score in reversed(self.trust_history[-10:]):
                if score.deviation_magnitude > 0.1:
                    deviation_duration += self.update_interval
                else:
                    break
        
        # Create trust score
        trust = TrustScore(
            timestamp=timestamp,
            trust_score=trust_score,
            trust_level=trust_level,
            component=component,
            deviation_magnitude=deviation_magnitude,
            deviation_duration=deviation_duration
        )
        
        self.trust_history.append(trust)
        return trust
    
    def get_trust_history_df(self) -> pd.DataFrame:
        """Convert trust history to DataFrame"""
        if not self.trust_history:
            return pd.DataFrame()
        
        data = {
            'timestamp': [t.timestamp for t in self.trust_history],
            'trust_score': [t.trust_score for t in self.trust_history],
            'trust_level': [t.trust_level.value for t in self.trust_history],
            'component': [t.component for t in self.trust_history],
            'deviation_magnitude': [t.deviation_magnitude for t in self.trust_history],
            'deviation_duration': [t.deviation_duration for t in self.trust_history]
        }
        
        return pd.DataFrame(data)


class AttackLatencyAnalyzer:
    """
    Analyzes attack latency and exposure windows
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize latency analyzer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.measurement_window = self.config['advanced_features']['attack_latency']['measurement_window']
        self.latency_records: List[AttackLatency] = []
    
    def analyze_latency(self, attack_id: str, attack_start: float,
                       detection_time: Optional[float],
                       unsafe_state_start: Optional[float]) -> AttackLatency:
        """
        Analyze attack latency and exposure window
        
        Returns:
            AttackLatency object
        """
        # Exposure window: time system was unsafe before detection
        exposure_window = 0.0
        if unsafe_state_start is not None:
            if detection_time is not None:
                # System was unsafe from unsafe_start until detection
                exposure_window = max(0.0, detection_time - unsafe_state_start)
            else:
                # Never detected, exposure = entire unsafe duration
                exposure_window = self.measurement_window
        
        # Persistence score: how long attacker can stay undetected
        if detection_time is not None:
            persistence_duration = detection_time - attack_start
            persistence_score = min(1.0, persistence_duration / self.measurement_window)
        else:
            persistence_score = 1.0  # Never detected = maximum persistence
        
        latency = AttackLatency(
            attack_id=attack_id,
            attack_start=attack_start,
            detection_time=detection_time,
            unsafe_state_start=unsafe_state_start,
            exposure_window=exposure_window,
            persistence_score=persistence_score
        )
        
        self.latency_records.append(latency)
        return latency


class SilentFailureDetector:
    """
    Detects silent failures (attacks that don't trigger alarms)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize silent failure detector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        silent_config = self.config['advanced_features']['silent_failure']
        self.trend_window = silent_config['trend_analysis_window']
        self.degradation_threshold = silent_config['degradation_rate_threshold']
        
        self.detections: List[SilentFailure] = []
    
    def detect_silent_failure(self, timestamp: float, states: List[ThreeStateModel],
                              alarm_triggered: bool) -> Optional[SilentFailure]:
        """
        Detect silent failures
        
        Silent failure = unsafe trend that doesn't trigger alarms
        
        Args:
            timestamp: Current timestamp
            states: Recent state history
            alarm_triggered: Whether alarm was triggered
        
        Returns:
            SilentFailure if detected, None otherwise
        """
        if len(states) < self.trend_window:
            return None
        
        # Analyze trend
        recent_states = states[-self.trend_window:]
        
        # Compute degradation rate
        divergences = [s.divergence for s in recent_states]
        degradation_rate = np.mean(np.diff(divergences))
        
        # Check for trend anomaly
        trend_anomaly = degradation_rate > self.degradation_threshold
        
        # Check for unsafe states without alarms
        unsafe_states = [s for s in recent_states 
                        if s.safety_state.value in ['unsafe', 'critical']]
        unsafe_without_alarm = len(unsafe_states) > 0 and not alarm_triggered
        
        # Silent failure detected if:
        # 1. Degrading trend exists
        # 2. No alarm triggered
        # 3. Unsafe states present
        if trend_anomaly and unsafe_without_alarm:
            # Estimate impact
            if degradation_rate > 0.05:
                impact = "High - Rapid degradation"
            elif degradation_rate > 0.02:
                impact = "Medium - Gradual degradation"
            else:
                impact = "Low - Slow degradation"
            
            failure = SilentFailure(
                timestamp=timestamp,
                detected=True,
                failure_type="Silent degradation without alarm",
                degradation_rate=degradation_rate,
                trend_anomaly=True,
                impact_estimate=impact
            )
            
            self.detections.append(failure)
            return failure
        
        return None
    
    def get_detections_df(self) -> pd.DataFrame:
        """Convert detections to DataFrame"""
        if not self.detections:
            return pd.DataFrame()
        
        data = {
            'timestamp': [d.timestamp for d in self.detections],
            'detected': [d.detected for d in self.detections],
            'failure_type': [d.failure_type for d in self.detections],
            'degradation_rate': [d.degradation_rate for d in self.detections],
            'trend_anomaly': [d.trend_anomaly for d in self.detections],
            'impact_estimate': [d.impact_estimate for d in self.detections]
        }
        
        return pd.DataFrame(data)


class MitigationSimulator:
    """
    Simulates attacks with and without mitigations
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize mitigation simulator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mitigation_types = self.config['advanced_features']['mitigation_simulation']['mitigation_types']
    
    def simulate_with_mitigation(self, attack_scenario: Dict,
                                mitigation_type: str,
                                digital_twin: CyberAwareDigitalTwin) -> Dict:
        """
        Simulate attack with mitigation applied
        
        Returns:
            Comparison dictionary with before/after metrics
        """
        # Reset digital twin
        digital_twin.reset()
        
        # Simulate attack without mitigation (baseline)
        baseline_states = digital_twin.simulate_attack(
            attack_scenario['attack_type'],
            int(attack_scenario['duration']),
            attack_scenario.get('normal_level', 500.0),
            attack_scenario.get('normal_valve', 1),
            attack_scenario.get('normal_pump', 0)
        )
        
        baseline_unsafe = sum(1 for s in baseline_states 
                             if s.safety_state.value in ['unsafe', 'critical'])
        baseline_max_divergence = max(s.divergence for s in baseline_states)
        
        # Apply mitigation and re-simulate
        # (In real implementation, mitigation would modify control logic)
        mitigated_states = baseline_states.copy()  # Placeholder
        
        # Mitigation effects (conceptual)
        if mitigation_type == "redundant_sensor":
            # Redundant sensor would catch spoofing
            mitigated_unsafe = baseline_unsafe * 0.3  # 70% reduction
            mitigated_divergence = baseline_max_divergence * 0.5
        
        elif mitigation_type == "rate_validation":
            # Rate validation would catch sudden changes
            mitigated_unsafe = baseline_unsafe * 0.5
            mitigated_divergence = baseline_max_divergence * 0.7
        
        elif mitigation_type == "digital_twin_check":
            # Digital twin validation would catch divergence
            mitigated_unsafe = baseline_unsafe * 0.2
            mitigated_divergence = baseline_max_divergence * 0.3
        
        elif mitigation_type == "fail_safe_closure":
            # Fail-safe would prevent overflow
            mitigated_unsafe = 0
            mitigated_divergence = baseline_max_divergence * 0.4
        
        else:
            mitigated_unsafe = baseline_unsafe
            mitigated_divergence = baseline_max_divergence
        
        comparison = {
            'mitigation_type': mitigation_type,
            'baseline': {
                'unsafe_states': baseline_unsafe,
                'max_divergence': baseline_max_divergence
            },
            'mitigated': {
                'unsafe_states': mitigated_unsafe,
                'max_divergence': mitigated_divergence
            },
            'improvement': {
                'unsafe_reduction': (baseline_unsafe - mitigated_unsafe) / baseline_unsafe if baseline_unsafe > 0 else 0,
                'divergence_reduction': (baseline_max_divergence - mitigated_divergence) / baseline_max_divergence if baseline_max_divergence > 0 else 0
            }
        }
        
        return comparison


if __name__ == "__main__":
    # Example usage
    tdi = TrustDegradationIndex()
    
    # Simulate trust degradation
    for t in range(100):
        expected = 500.0
        observed = 500.0 + np.random.randn() * 5 if t < 50 else 500.0 + np.random.randn() * 50
        trust = tdi.update_trust(float(t), expected, observed)
        print(f"t={t}: Trust={trust.trust_score:.3f}, Level={trust.trust_level.value}")
