"""
Attack Scenario Generator

Generates and simulates cyberattack scenarios on the digital twin
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import yaml
from datetime import datetime
import json
from pathlib import Path

from .digital_twin import CyberAwareDigitalTwin
from .genai_engine import LSTMAutoencoder


class AttackGenerator:
    """
    Generates and simulates attack scenarios
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize attack generator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.attack_types = self.config['genai']['attack_generation']['attack_types']
        self.num_synthetic_attacks = self.config['genai']['attack_generation']['num_synthetic_attacks']
        
        self.digital_twin = CyberAwareDigitalTwin(config_path)
        self.attack_scenarios: List[Dict] = []
        
    def generate_sensor_spoofing_attack(self, duration: int = 60,
                                       normal_level: float = 500.0,
                                       spoof_factor: float = 0.3) -> Dict:
        """
        Generate sensor spoofing attack
        
        Attack: Sensor reports false low level to keep valve open
        """
        self.digital_twin.reset()
        
        # Normal operation first
        for t in range(10):
            self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=normal_level + t * 0.5,
                inlet_valve=1,
                outlet_pump=0
            )
        
        # Attack phase
        attack_start = 10.0
        attack_states = []
        
        for t in range(10, 10 + duration):
            # Actual level continues to rise (valve still open)
            actual_level = normal_level + t * 0.5
            
            # But sensor reports false low level
            spoofed_level = actual_level * spoof_factor
            
            state = self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=spoofed_level,  # Spoofed reading
                inlet_valve=1,  # Valve stays open (controller thinks level is low)
                outlet_pump=0
            )
            attack_states.append(state)
        
        # Post-attack
        for t in range(10 + duration, 10 + duration + 10):
            self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=normal_level + t * 0.5,
                inlet_valve=1,
                outlet_pump=0
            )
        
        scenario = {
            'attack_id': f"sensor_spoofing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'attack_type': 'sensor_spoofing',
            'start_time': attack_start,
            'end_time': float(10 + duration),
            'duration': float(duration),
            'description': 'Sensor reports false low level to keep inlet valve open, causing overflow risk',
            'states': attack_states,
            'digital_twin_history': self.digital_twin.get_state_history()
        }
        
        self.attack_scenarios.append(scenario)
        return scenario
    
    def generate_slow_manipulation_attack(self, duration: int = 120,
                                         normal_level: float = 500.0,
                                         drift_rate: float = 0.01) -> Dict:
        """
        Generate slow manipulation attack (stealthy)
        
        Attack: Gradual sensor drift that doesn't trigger immediate alarms
        """
        self.digital_twin.reset()
        
        # Normal operation
        for t in range(20):
            self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=normal_level + np.random.normal(0, 5),
                inlet_valve=1,
                outlet_pump=0
            )
        
        # Attack phase - gradual drift
        attack_start = 20.0
        attack_states = []
        
        for t in range(20, 20 + duration):
            # Actual level
            actual_level = normal_level + t * 0.3
            
            # Sensor drifts gradually
            drift = (t - 20) * drift_rate
            manipulated_level = actual_level * (1.0 - drift)
            
            state = self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=manipulated_level,
                inlet_valve=1,
                outlet_pump=0
            )
            attack_states.append(state)
        
        scenario = {
            'attack_id': f"slow_manipulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'attack_type': 'slow_manipulation',
            'start_time': attack_start,
            'end_time': float(20 + duration),
            'duration': float(duration),
            'description': 'Gradual sensor manipulation that evades immediate detection',
            'states': attack_states,
            'digital_twin_history': self.digital_twin.get_state_history()
        }
        
        self.attack_scenarios.append(scenario)
        return scenario
    
    def generate_frozen_sensor_attack(self, duration: int = 60,
                                      normal_level: float = 500.0) -> Dict:
        """
        Generate frozen sensor attack
        
        Attack: Sensor stuck at one value
        """
        self.digital_twin.reset()
        
        # Normal operation
        for t in range(10):
            self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=normal_level + t * 0.5,
                inlet_valve=1,
                outlet_pump=0
            )
        
        # Attack phase - sensor frozen
        attack_start = 10.0
        frozen_value = normal_level + 10 * 0.5  # Freeze at value from step 10
        attack_states = []
        
        for t in range(10, 10 + duration):
            # Actual level continues to change
            actual_level = normal_level + t * 0.5
            
            # But sensor reports frozen value
            state = self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=frozen_value,  # Frozen
                inlet_valve=1,
                outlet_pump=0
            )
            attack_states.append(state)
        
        scenario = {
            'attack_id': f"frozen_sensor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'attack_type': 'frozen_sensor',
            'start_time': attack_start,
            'end_time': float(10 + duration),
            'duration': float(duration),
            'description': 'Sensor frozen at constant value while actual level changes',
            'states': attack_states,
            'digital_twin_history': self.digital_twin.get_state_history()
        }
        
        self.attack_scenarios.append(scenario)
        return scenario
    
    def generate_delayed_response_attack(self, duration: int = 60,
                                        normal_level: float = 500.0,
                                        delay_steps: int = 5) -> Dict:
        """
        Generate delayed response attack
        
        Attack: Sensor reports values from previous time steps
        """
        self.digital_twin.reset()
        
        # Normal operation
        level_history = []
        for t in range(20):
            level = normal_level + t * 0.5
            level_history.append(level)
            self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=level,
                inlet_valve=1,
                outlet_pump=0
            )
        
        # Attack phase - delayed response
        attack_start = 20.0
        attack_states = []
        
        for t in range(20, 20 + duration):
            # Actual level
            actual_level = normal_level + t * 0.5
            level_history.append(actual_level)
            
            # Sensor reports delayed value
            delayed_idx = max(0, len(level_history) - delay_steps - 1)
            delayed_level = level_history[delayed_idx]
            
            state = self.digital_twin.update_state(
                timestamp=float(t),
                level_sensor=delayed_level,  # Delayed
                inlet_valve=1,
                outlet_pump=0
            )
            attack_states.append(state)
        
        scenario = {
            'attack_id': f"delayed_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'attack_type': 'delayed_response',
            'start_time': attack_start,
            'end_time': float(20 + duration),
            'duration': float(duration),
            'description': 'Sensor reports delayed values, causing controller to react to outdated information',
            'states': attack_states,
            'digital_twin_history': self.digital_twin.get_state_history()
        }
        
        self.attack_scenarios.append(scenario)
        return scenario
    
    def generate_all_attack_scenarios(self) -> List[Dict]:
        """Generate all predefined attack scenarios"""
        scenarios = []
        
        # Sensor spoofing
        scenarios.append(self.generate_sensor_spoofing_attack(duration=60))
        
        # Slow manipulation
        scenarios.append(self.generate_slow_manipulation_attack(duration=120))
        
        # Frozen sensor
        scenarios.append(self.generate_frozen_sensor_attack(duration=60))
        
        # Delayed response
        scenarios.append(self.generate_delayed_response_attack(duration=60))
        
        return scenarios
    
    def save_attack_scenario(self, scenario: Dict, filepath: str):
        """Save attack scenario to JSON"""
        # Convert states to serializable format
        export_scenario = scenario.copy()
        
        # Convert state history DataFrame to dict
        if 'digital_twin_history' in export_scenario:
            df = export_scenario['digital_twin_history']
            if isinstance(df, pd.DataFrame):
                export_scenario['digital_twin_history'] = df.to_dict('records')
        
        # Remove non-serializable states
        if 'states' in export_scenario:
            export_scenario['states'] = [
                {
                    'expected_state': s.expected_state,
                    'observed_state': s.observed_state,
                    'believed_state': s.believed_state,
                    'divergence': s.divergence,
                    'safety_state': s.safety_state.value
                }
                for s in export_scenario['states']
            ]
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_scenario, f, indent=2)
    
    def load_attack_scenario(self, filepath: str) -> Dict:
        """Load attack scenario from JSON"""
        with open(filepath, 'r') as f:
            scenario = json.load(f)
        return scenario


if __name__ == "__main__":
    # Example usage
    generator = AttackGenerator()
    
    # Generate sensor spoofing attack
    scenario = generator.generate_sensor_spoofing_attack(duration=60)
    print(f"Generated attack: {scenario['attack_id']}")
    print(f"Attack type: {scenario['attack_type']}")
    print(f"Duration: {scenario['duration']} seconds")
    
    # Save scenario
    generator.save_attack_scenario(
        scenario,
        f"outputs/results/{scenario['attack_id']}.json"
    )
