"""
Enhanced Digital Twin with Physics Coupling

Models system dynamics with state evolution and safety boundary tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import yaml
from scipy.integrate import odeint


class SafetyBoundary(Enum):
    """Safety boundary states"""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    CRITICAL = "critical"


@dataclass
class PhysicsState:
    """Physics-based state representation"""
    timestamp: float
    level: float  # mm
    flow_in: float  # mm/s
    flow_out: float  # mm/s
    net_flow: float  # mm/s
    rate_of_change: float  # mm/s
    distance_to_boundary: float  # mm
    safety_margin: float  # [0, 1]
    safety_boundary: SafetyBoundary


class PhysicsCoupledDigitalTwin:
    """
    Digital twin with physics-based state evolution modeling
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize physics-coupled digital twin"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.thresholds = self.config['subsystem']['thresholds']
        self.control = self.config['subsystem']['control']
        
        # Physics parameters
        self.tank_capacity = 1000.0  # mm
        self.tank_area = 1.0  # m^2 (assumed)
        self.gravity = 9.81  # m/s^2
        
        # State history
        self.state_history: List[PhysicsState] = []
        self.safety_boundary_history: List[SafetyBoundary] = []
        
    def compute_physics_state(self, timestamp: float, level: float,
                            inlet_valve: int, outlet_pump: int,
                            dt: float = 1.0) -> PhysicsState:
        """
        Compute physics-based state evolution
        
        Args:
            timestamp: Current time
            level: Current tank level (mm)
            inlet_valve: Valve state (0 or 1)
            outlet_pump: Pump state (0 or 1)
            dt: Time step (seconds)
        
        Returns:
            PhysicsState object
        """
        # Compute flows
        flow_in = self.control['valve_flow_rate'] if inlet_valve == 1 else 0.0
        flow_out = self.control['pump_flow_rate'] if outlet_pump == 1 else 0.0
        net_flow = flow_in - flow_out
        
        # Compute rate of change
        if len(self.state_history) > 0:
            prev_level = self.state_history[-1].level
            rate_of_change = (level - prev_level) / dt
        else:
            rate_of_change = net_flow
        
        # Compute distance to safety boundaries
        distance_to_max = self.thresholds['max_level'] - level
        distance_to_min = level - self.thresholds['min_level']
        distance_to_boundary = min(distance_to_max, distance_to_min)
        
        # Compute safety margin [0, 1]
        safe_range = self.thresholds['max_level'] - self.thresholds['min_level']
        if safe_range > 0:
            safety_margin = distance_to_boundary / (safe_range / 2)
            safety_margin = np.clip(safety_margin, 0.0, 1.0)
        else:
            safety_margin = 0.0
        
        safety_boundary = self._assess_safety_boundary(level, rate_of_change)

        state = PhysicsState(
            timestamp=timestamp,
            level=level,
            flow_in=flow_in,
            flow_out=flow_out,
            net_flow=net_flow,
            rate_of_change=rate_of_change,
            distance_to_boundary=distance_to_boundary,
            safety_margin=safety_margin,
            safety_boundary=safety_boundary
        )


        
        self.state_history.append(state)
        
        # Assess safety boundary
        safety_boundary = self._assess_safety_boundary(level, rate_of_change)
        self.safety_boundary_history.append(safety_boundary)
        
        return state
    
    def _assess_safety_boundary(self, level: float, rate_of_change: float) -> SafetyBoundary:
        """
        Assess safety boundary based on level and rate of change
        
        Args:
            level: Current level
            rate_of_change: Rate of level change
        
        Returns:
            SafetyBoundary enum
        """
        # Check critical boundaries
        if level >= self.thresholds['critical_overflow']:
            return SafetyBoundary.CRITICAL
        if level <= self.thresholds['critical_dry']:
            return SafetyBoundary.CRITICAL
        
        # Check unsafe boundaries
        if level >= self.thresholds['max_level']:
            return SafetyBoundary.UNSAFE
        if level <= self.thresholds['min_level']:
            return SafetyBoundary.UNSAFE
        
        # Check warning zones (approaching boundaries)
        warning_zone_max = self.thresholds['max_level'] * 0.9
        warning_zone_min = self.thresholds['min_level'] * 1.1
        
        if level >= warning_zone_max or level <= warning_zone_min:
            # Also check if rate is pushing toward boundary
            if (level >= warning_zone_max and rate_of_change > 0) or \
               (level <= warning_zone_min and rate_of_change < 0):
                return SafetyBoundary.WARNING
        
        return SafetyBoundary.SAFE
    
    def predict_state_evolution(self, current_state: PhysicsState,
                               control_inputs: Dict,
                               prediction_horizon: float = 10.0,
                               dt: float = 1.0) -> List[PhysicsState]:
        """
        Predict future state evolution
        
        Args:
            current_state: Current physics state
            control_inputs: Future control inputs (valve, pump)
            prediction_horizon: Time to predict ahead (seconds)
            dt: Time step
        
        Returns:
            List of predicted states
        """
        predicted_states = []
        current_level = current_state.level
        
        n_steps = int(prediction_horizon / dt)
        
        for i in range(n_steps):
            t = current_state.timestamp + (i + 1) * dt
            
            # Get control inputs (assume constant if not specified)
            inlet_valve = control_inputs.get('inlet_valve', 1)
            outlet_pump = control_inputs.get('outlet_pump', 0)
            
            # Compute next state
            flow_in = self.control['valve_flow_rate'] if inlet_valve == 1 else 0.0
            flow_out = self.control['pump_flow_rate'] if outlet_pump == 1 else 0.0
            net_flow = flow_in - flow_out
            
            # Update level
            current_level = current_level + net_flow * dt
            current_level = np.clip(current_level, 0.0, self.tank_capacity)
            
            # Create predicted state
            predicted_state = PhysicsState(
                timestamp=t,
                level=current_level,
                flow_in=flow_in,
                flow_out=flow_out,
                net_flow=net_flow,
                rate_of_change=net_flow,
                distance_to_boundary=min(
                    self.thresholds['max_level'] - current_level,
                    current_level - self.thresholds['min_level']
                ),
                safety_margin=0.0  # Would compute if needed
            )
            
            predicted_states.append(predicted_state)
        
        return predicted_states
    
    def identify_unsafe_trajectory(self, predicted_states: List[PhysicsState]) -> Tuple[bool, Optional[float]]:
        """
        Identify if predicted trajectory leads to unsafe state
        
        Args:
            predicted_states: List of predicted states
        
        Returns:
            (is_unsafe, time_to_unsafe) - Time to unsafe state in seconds
        """
        for i, state in enumerate(predicted_states):
            safety = self._assess_safety_boundary(state.level, state.rate_of_change)
            if safety in [SafetyBoundary.UNSAFE, SafetyBoundary.CRITICAL]:
                time_to_unsafe = i * 1.0  # Assuming 1s time step
                return True, time_to_unsafe
        
        return False, None
    
    def compute_state_divergence_physics(self, expected_state: PhysicsState,
                                        observed_level: float) -> float:
        """
        Compute physics-consistent divergence
        
        Args:
            expected_state: Expected physics state
            observed_level: Observed sensor level
        
        Returns:
            Divergence score [0, 1]
        """
        # Physical consistency check
        level_diff = abs(expected_state.level - observed_level)
        
        # Normalize by safe range
        safe_range = self.thresholds['max_level'] - self.thresholds['min_level']
        if safe_range > 0:
            normalized_divergence = level_diff / safe_range
        else:
            normalized_divergence = 1.0
        
        # Also check if divergence violates physics (e.g., level can't decrease if valve closed and pump off)
        physics_violation = False
        if expected_state.flow_in == 0 and expected_state.flow_out == 0:
            # No flow, level should be constant
            if abs(observed_level - expected_state.level) > 5.0:  # 5mm tolerance
                physics_violation = True
        
        if physics_violation:
            normalized_divergence = max(normalized_divergence, 0.5)  # Boost divergence
        
        return np.clip(normalized_divergence, 0.0, 1.0)
    
    def get_state_history_df(self) -> pd.DataFrame:
        """Convert state history to DataFrame"""
        if not self.state_history:
            return pd.DataFrame()
        
        data = {
            'timestamp': [s.timestamp for s in self.state_history],
            'level': [s.level for s in self.state_history],
            'flow_in': [s.flow_in for s in self.state_history],
            'flow_out': [s.flow_out for s in self.state_history],
            'net_flow': [s.net_flow for s in self.state_history],
            'rate_of_change': [s.rate_of_change for s in self.state_history],
            'distance_to_boundary': [s.distance_to_boundary for s in self.state_history],
            'safety_margin': [s.safety_margin for s in self.state_history],
            'safety_boundary': [s.value for s in self.safety_boundary_history]
        }
        
        return pd.DataFrame(data)
    
    def reset(self):
        """Reset digital twin state"""
        self.state_history = []
        self.safety_boundary_history = []


if __name__ == "__main__":
    # Example usage
    twin = PhysicsCoupledDigitalTwin()
    
    # Simulate state evolution
    for t in range(100):
        level = 500.0 + t * 2.0  # Rising level
        state = twin.compute_physics_state(float(t), level, 1, 0)
        
        if t % 20 == 0:
            print(f"t={t}: Level={state.level:.1f}mm, "
                  f"Safety={state.safety_margin:.2f}, "
                  f"Boundary={twin.safety_boundary_history[-1].value}")
