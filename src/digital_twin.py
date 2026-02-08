"""
Cyber-Aware Digital Twin for Raw Water Tank Level Control System

Models THREE STATES:
1. Expected State (what should happen based on control logic)
2. Observed State (what sensors actually report)
3. Believed State (what controller thinks is happening)

Cyberattack = Divergence between these states
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from dataclasses import dataclass
from enum import Enum


class SafetyState(Enum):
    """Safety state classification"""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    CRITICAL = "critical"


@dataclass
class TankState:
    """Represents tank state at a point in time"""
    timestamp: float
    level: float  # mm
    inlet_valve: int  # 0 or 1
    outlet_pump: int  # 0 or 1
    level_sensor: float  # mm (observed)


@dataclass
class ThreeStateModel:
    """Three-state model for cyber-aware analysis"""
    expected_state: float  # What digital twin predicts
    observed_state: float  # What sensor reports
    believed_state: float  # What controller thinks
    divergence: float  # Measure of cyber-physical divergence
    safety_state: SafetyState


class CyberAwareDigitalTwin:
    """
    Digital Twin that models expected behavior and detects cyber-physical divergence
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize digital twin with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.thresholds = self.config['subsystem']['thresholds']
        self.control = self.config['subsystem']['control']
        self.divergence_threshold = self.config['digital_twin']['state_comparison']['divergence_threshold']
        self.rate_threshold = self.config['digital_twin']['state_comparison']['rate_of_change_threshold']
        
        # State history
        self.state_history: List[ThreeStateModel] = []
        self.tank_state_history: List[TankState] = []
        
    def predict_expected_level(self, current_level: float, 
                              inlet_valve: int, outlet_pump: int,
                              dt: float = 1.0) -> float:
        """
        Predict expected tank level based on control logic
        
        Simple rule-based model:
        - IF inlet valve open → level increases
        - IF outlet pump ON → level decreases
        """
        # Flow rates (mm/s)
        valve_flow = self.control['valve_flow_rate'] if inlet_valve == 1 else 0.0
        pump_flow = self.control['pump_flow_rate'] if outlet_pump == 1 else 0.0
        
        # Net flow
        net_flow = (valve_flow - pump_flow) * dt
        
        # Expected level
        expected_level = current_level + net_flow
        
        # Physical constraints (tank has limits)
        expected_level = max(0.0, min(expected_level, 1000.0))
        
        return expected_level
    
    def compute_state_divergence(self, expected: float, observed: float, 
                                 believed: float) -> float:
        """
        Compute divergence between three states
        
        Divergence = max(|expected - observed|, |expected - believed|, |observed - believed|)
        Normalized by expected value
        """
        if expected == 0:
            expected = 1.0  # Avoid division by zero
        
        divergence_expected_observed = abs(expected - observed) / expected
        divergence_expected_believed = abs(expected - believed) / expected
        divergence_observed_believed = abs(observed - believed) / expected
        
        max_divergence = max(
            divergence_expected_observed,
            divergence_expected_believed,
            divergence_observed_believed
        )
        
        return max_divergence
    
    def assess_safety_state(self, level: float) -> SafetyState:
        """
        Assess safety state based on tank level
        """
        if level >= self.thresholds['critical_overflow'] or level <= self.thresholds['critical_dry']:
            return SafetyState.CRITICAL
        elif level >= self.thresholds['max_level'] or level <= self.thresholds['min_level']:
            return SafetyState.UNSAFE
        elif (level >= self.thresholds['max_level'] * 0.9 or 
              level <= self.thresholds['min_level'] * 1.1):
            return SafetyState.WARNING
        else:
            return SafetyState.SAFE
    
    def check_rate_of_change(self, current_level: float, previous_level: float,
                            dt: float = 1.0) -> Tuple[float, bool]:
        """
        Check rate of change of tank level
        Returns: (rate, is_anomalous)
        """
        rate = abs(current_level - previous_level) / dt
        
        is_anomalous = rate > self.rate_threshold
        
        return rate, is_anomalous
    
    def update_state(self, timestamp: float, level_sensor: float,
                    inlet_valve: int, outlet_pump: int,
                    previous_expected_level: Optional[float] = None) -> ThreeStateModel:
        """
        Update three-state model for current time step
        
        Args:
            timestamp: Current time
            level_sensor: Observed sensor reading
            inlet_valve: Valve state (0 or 1)
            outlet_pump: Pump state (0 or 1)
            previous_expected_level: Previous expected level (for prediction)
        
        Returns:
            ThreeStateModel with all three states and divergence
        """
        # Get previous expected level from history or use current observed
        if previous_expected_level is None:
            if self.tank_state_history:
                prev_state = self.tank_state_history[-1]
                prev_expected = prev_state.level  # Use previous expected
            else:
                prev_expected = level_sensor  # First step: use observed
        else:
            prev_expected = previous_expected_level
        
        # Compute expected state (what should happen)
        expected_level = self.predict_expected_level(
            prev_expected, inlet_valve, outlet_pump
        )
        
        # Observed state (what sensor reports)
        observed_level = level_sensor
        
        # Believed state (what controller thinks - typically same as observed unless attack)
        # In normal operation, controller believes what sensor says
        # In attack, controller may be deceived
        believed_level = observed_level  # Default: controller trusts sensor
        
        # Compute divergence
        divergence = self.compute_state_divergence(
            expected_level, observed_level, believed_level
        )
        
        # Assess safety
        safety_state = self.assess_safety_state(observed_level)
        
        # Create three-state model
        three_state = ThreeStateModel(
            expected_state=expected_level,
            observed_state=observed_level,
            believed_state=believed_level,
            divergence=divergence,
            safety_state=safety_state
        )
        
        # Store state
        tank_state = TankState(
            timestamp=timestamp,
            level=expected_level,  # Store expected as "true" state
            inlet_valve=inlet_valve,
            outlet_pump=outlet_pump,
            level_sensor=observed_level
        )
        
        self.state_history.append(three_state)
        self.tank_state_history.append(tank_state)
        
        return three_state
    
    def detect_cyber_anomaly(self, three_state: ThreeStateModel) -> bool:
        """
        Detect if current state indicates cyber anomaly
        
        Anomaly = divergence exceeds threshold OR safety state is unsafe
        """
        is_divergence_anomaly = three_state.divergence > self.divergence_threshold
        is_safety_anomaly = three_state.safety_state in [
            SafetyState.UNSAFE, SafetyState.CRITICAL
        ]
        
        return is_divergence_anomaly or is_safety_anomaly
    
    def simulate_attack(self, attack_type: str, duration: int,
                       normal_level: float, normal_valve: int, normal_pump: int) -> List[ThreeStateModel]:
        """
        Simulate cyberattack on the digital twin
        
        Attack types:
        - sensor_spoofing: Sensor reports false value
        - slow_manipulation: Gradual sensor drift
        - frozen_sensor: Sensor stuck at one value
        - delayed_response: Sensor reports delayed values
        """
        states = []
        
        for t in range(duration):
            timestamp = float(t)
            
            if attack_type == "sensor_spoofing":
                # Sensor reports false low level (to keep valve open)
                spoofed_level = normal_level * 0.3  # Report 30% of actual
                observed = spoofed_level
                # Controller believes the spoofed value
                believed = spoofed_level
                
            elif attack_type == "slow_manipulation":
                # Gradual drift
                drift_factor = 1.0 - (t / duration) * 0.5  # Slowly decrease
                observed = normal_level * drift_factor
                believed = observed
                
            elif attack_type == "frozen_sensor":
                # Sensor stuck at initial value
                observed = normal_level
                believed = observed
                
            elif attack_type == "delayed_response":
                # Sensor reports value from 5 steps ago
                delay = 5
                if t >= delay and self.tank_state_history:
                    observed = self.tank_state_history[t - delay].level_sensor
                else:
                    observed = normal_level
                believed = observed
                
            else:
                observed = normal_level
                believed = observed
            
            # Update state
            prev_expected = self.tank_state_history[-1].level if self.tank_state_history else normal_level
            state = self.update_state(
                timestamp, observed, normal_valve, normal_pump, prev_expected
            )
            states.append(state)
        
        return states
    
    def get_state_history(self) -> pd.DataFrame:
        """Convert state history to DataFrame for analysis"""
        if not self.state_history:
            return pd.DataFrame()
        
        data = {
            'timestamp': [i for i in range(len(self.state_history))],
            'expected_state': [s.expected_state for s in self.state_history],
            'observed_state': [s.observed_state for s in self.state_history],
            'believed_state': [s.believed_state for s in self.state_history],
            'divergence': [s.divergence for s in self.state_history],
            'safety_state': [s.safety_state.value for s in self.state_history]
        }
        
        return pd.DataFrame(data)
    
    def reset(self):
        """Reset digital twin state"""
        self.state_history = []
        self.tank_state_history = []


if __name__ == "__main__":
    # Example usage
    twin = CyberAwareDigitalTwin()
    
    # Simulate normal operation
    for t in range(10):
        state = twin.update_state(
            timestamp=float(t),
            level_sensor=500.0 + t * 2.0,  # Gradual increase
            inlet_valve=1,
            outlet_pump=0
        )
        print(f"t={t}: Expected={state.expected_state:.2f}, "
              f"Observed={state.observed_state:.2f}, "
              f"Divergence={state.divergence:.4f}, "
              f"Safety={state.safety_state.value}")
