"""
Defensive Decision Support Engine

Provides attack-specific mitigation recommendations and defensive intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import yaml


class MitigationPriority(Enum):
    """Mitigation priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DefensiveAction(Enum):
    """Defensive action types"""
    VALIDATE_SENSOR = "validate_sensor"
    ENABLE_REDUNDANCY = "enable_redundancy"
    CLOSE_VALVE_SAFELY = "close_valve_safely"
    STOP_PUMP = "stop_pump"
    ENABLE_RATE_MONITORING = "enable_rate_monitoring"
    CROSS_SENSOR_CHECK = "cross_sensor_check"
    DIGITAL_TWIN_VALIDATION = "digital_twin_validation"
    ISOLATE_SUBSYSTEM = "isolate_subsystem"
    ALERT_OPERATOR = "alert_operator"
    LOG_INCIDENT = "log_incident"


@dataclass
class MitigationRecommendation:
    """Structured mitigation recommendation"""
    action: DefensiveAction
    priority: MitigationPriority
    attack_type: str
    rationale: str
    expected_effectiveness: float  # [0, 1]
    implementation_time: float  # seconds
    risk_reduction: float  # [0, 1]
    description: str


class DefensiveDecisionSupport:
    """
    Provides defensive decision support for detected attacks
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize defensive decision support"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.recommendations: List[MitigationRecommendation] = []
        self.action_history: List[Dict] = []
    
    def generate_mitigations(self, attack_type: str,
                           anomaly_explanation: Dict,
                           digital_twin_state: Dict,
                           risk_score: float) -> List[MitigationRecommendation]:
        """
        Generate attack-specific mitigation recommendations
        
        Args:
            attack_type: Type of detected attack
            anomaly_explanation: Explanation from explainability engine
            digital_twin_state: Current digital twin state
            risk_score: Overall risk score [0, 1]
        
        Returns:
            List of prioritized mitigation recommendations
        """
        mitigations = []
        
        # Attack-specific mitigations
        if attack_type == "sensor_spoofing":
            mitigations.extend(self._mitigate_sensor_spoofing(
                anomaly_explanation, digital_twin_state, risk_score
            ))
        elif attack_type == "replay_attack":
            mitigations.extend(self._mitigate_replay_attack(
                anomaly_explanation, digital_twin_state, risk_score
            ))
        elif attack_type == "gradual_manipulation":
            mitigations.extend(self._mitigate_gradual_manipulation(
                anomaly_explanation, digital_twin_state, risk_score
            ))
        elif attack_type == "frozen_sensor":
            mitigations.extend(self._mitigate_frozen_sensor(
                anomaly_explanation, digital_twin_state, risk_score
            ))
        elif attack_type == "delay_dos":
            mitigations.extend(self._mitigate_delay_dos(
                anomaly_explanation, digital_twin_state, risk_score
            ))
        else:
            # Generic mitigations
            mitigations.extend(self._mitigate_generic(
                anomaly_explanation, digital_twin_state, risk_score
            ))
        
        # Sort by priority and effectiveness
        mitigations.sort(
            key=lambda m: (
                self._priority_weight(m.priority),
                -m.expected_effectiveness
            ),
            reverse=True
        )
        
        self.recommendations.extend(mitigations)
        return mitigations
    
    def _mitigate_sensor_spoofing(self, explanation: Dict,
                                  state: Dict, risk: float) -> List[MitigationRecommendation]:
        """Generate mitigations for sensor spoofing"""
        mitigations = []
        
        # Critical: Validate with redundant sensor
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.VALIDATE_SENSOR,
            priority=MitigationPriority.CRITICAL,
            attack_type="sensor_spoofing",
            rationale="Sensor spoofing detected - immediate validation required",
            expected_effectiveness=0.9,
            implementation_time=1.0,
            risk_reduction=0.8,
            description="Enable redundant sensor validation and compare readings"
        ))
        
        # High: Digital twin validation
        if state.get('divergence', 0) > 0.2:
            mitigations.append(MitigationRecommendation(
                action=DefensiveAction.DIGITAL_TWIN_VALIDATION,
                priority=MitigationPriority.HIGH,
                attack_type="sensor_spoofing",
                rationale="Large divergence between expected and observed states",
                expected_effectiveness=0.85,
                implementation_time=0.5,
                risk_reduction=0.7,
                description="Use digital twin prediction to validate sensor reading"
            ))
        
        # High: Fail-safe valve closure if overflow risk
        if state.get('safety_state') in ['unsafe', 'critical']:
            mitigations.append(MitigationRecommendation(
                action=DefensiveAction.CLOSE_VALVE_SAFELY,
                priority=MitigationPriority.CRITICAL,
                attack_type="sensor_spoofing",
                rationale="Unsafe state detected - prevent overflow",
                expected_effectiveness=1.0,
                implementation_time=2.0,
                risk_reduction=0.95,
                description="Immediately close inlet valve to prevent overflow"
            ))
        
        return mitigations
    
    def _mitigate_replay_attack(self, explanation: Dict,
                               state: Dict, risk: float) -> List[MitigationRecommendation]:
        """Generate mitigations for replay attack"""
        mitigations = []
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.CROSS_SENSOR_CHECK,
            priority=MitigationPriority.HIGH,
            attack_type="replay_attack",
            rationale="Replay attacks show temporal inconsistencies",
            expected_effectiveness=0.8,
            implementation_time=1.0,
            risk_reduction=0.7,
            description="Validate sensor readings against other sensors for temporal consistency"
        ))
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.ENABLE_RATE_MONITORING,
            priority=MitigationPriority.MEDIUM,
            attack_type="replay_attack",
            rationale="Detect sudden temporal jumps",
            expected_effectiveness=0.75,
            implementation_time=0.5,
            risk_reduction=0.6,
            description="Enable rate-of-change monitoring to detect replay patterns"
        ))
        
        return mitigations
    
    def _mitigate_gradual_manipulation(self, explanation: Dict,
                                      state: Dict, risk: float) -> List[MitigationRecommendation]:
        """Generate mitigations for gradual manipulation"""
        mitigations = []
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.ENABLE_RATE_MONITORING,
            priority=MitigationPriority.HIGH,
            attack_type="gradual_manipulation",
            rationale="Gradual drift requires trend analysis",
            expected_effectiveness=0.85,
            implementation_time=1.0,
            risk_reduction=0.75,
            description="Enable trend-based monitoring with predictive thresholds"
        ))
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.DIGITAL_TWIN_VALIDATION,
            priority=MitigationPriority.MEDIUM,
            attack_type="gradual_manipulation",
            rationale="Digital twin can detect gradual divergence",
            expected_effectiveness=0.8,
            implementation_time=0.5,
            risk_reduction=0.7,
            description="Use digital twin to track expected vs observed trends"
        ))
        
        return mitigations
    
    def _mitigate_frozen_sensor(self, explanation: Dict,
                               state: Dict, risk: float) -> List[MitigationRecommendation]:
        """Generate mitigations for frozen sensor"""
        mitigations = []
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.VALIDATE_SENSOR,
            priority=MitigationPriority.CRITICAL,
            attack_type="frozen_sensor",
            rationale="Sensor appears frozen - immediate validation required",
            expected_effectiveness=0.9,
            implementation_time=1.0,
            risk_reduction=0.85,
            description="Check sensor health and validate with redundant sensor"
        ))
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.ENABLE_RATE_MONITORING,
            priority=MitigationPriority.HIGH,
            attack_type="frozen_sensor",
            rationale="Zero rate-of-change indicates frozen sensor",
            expected_effectiveness=0.95,
            implementation_time=0.5,
            risk_reduction=0.9,
            description="Enable zero-change detection for sensor health monitoring"
        ))
        
        return mitigations
    
    def _mitigate_delay_dos(self, explanation: Dict,
                           state: Dict, risk: float) -> List[MitigationRecommendation]:
        """Generate mitigations for delay/DoS attack"""
        mitigations = []
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.CROSS_SENSOR_CHECK,
            priority=MitigationPriority.HIGH,
            attack_type="delay_dos",
            rationale="Delayed responses create temporal inconsistencies",
            expected_effectiveness=0.8,
            implementation_time=1.0,
            risk_reduction=0.7,
            description="Validate temporal alignment across sensors"
        ))
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.ISOLATE_SUBSYSTEM,
            priority=MitigationPriority.MEDIUM,
            attack_type="delay_dos",
            rationale="Prevent cascading effects from delayed responses",
            expected_effectiveness=0.7,
            implementation_time=5.0,
            risk_reduction=0.6,
            description="Isolate affected subsystem to prevent propagation"
        ))
        
        return mitigations
    
    def _mitigate_generic(self, explanation: Dict,
                         state: Dict, risk: float) -> List[MitigationRecommendation]:
        """Generate generic mitigations"""
        mitigations = []
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.ALERT_OPERATOR,
            priority=MitigationPriority.HIGH,
            attack_type="generic",
            rationale="Anomaly detected - operator attention required",
            expected_effectiveness=1.0,
            implementation_time=0.1,
            risk_reduction=0.5,
            description="Immediately alert operator with anomaly details"
        ))
        
        mitigations.append(MitigationRecommendation(
            action=DefensiveAction.LOG_INCIDENT,
            priority=MitigationPriority.MEDIUM,
            attack_type="generic",
            rationale="Document incident for analysis",
            expected_effectiveness=1.0,
            implementation_time=0.1,
            risk_reduction=0.1,
            description="Log incident details for forensic analysis"
        ))
        
        return mitigations
    
    def _priority_weight(self, priority: MitigationPriority) -> int:
        """Get numeric weight for priority"""
        weights = {
            MitigationPriority.CRITICAL: 4,
            MitigationPriority.HIGH: 3,
            MitigationPriority.MEDIUM: 2,
            MitigationPriority.LOW: 1
        }
        return weights.get(priority, 0)
    
    def generate_mitigation_table(self) -> pd.DataFrame:
        """
        Generate mitigation recommendations table
        
        Returns:
            DataFrame with mitigations
        """
        data = []
        for rec in self.recommendations:
            data.append({
                'Action': rec.action.value,
                'Priority': rec.priority.value,
                'Attack Type': rec.attack_type,
                'Effectiveness': f"{rec.expected_effectiveness:.1%}",
                'Risk Reduction': f"{rec.risk_reduction:.1%}",
                'Implementation Time': f"{rec.implementation_time:.1f}s",
                'Description': rec.description
            })
        
        return pd.DataFrame(data)
    
    def get_prioritized_actions(self, limit: int = 5) -> List[MitigationRecommendation]:
        """
        Get top N prioritized actions
        
        Args:
            limit: Maximum number of actions to return
        
        Returns:
            List of top recommendations
        """
        return self.recommendations[:limit]


if __name__ == "__main__":
    # Example usage
    support = DefensiveDecisionSupport()
    
    # Generate mitigations
    explanation = {
        'primary_cause': 'Level sensor shows significant deviation',
        'sensor_contributions': {'LIT101': 0.8}
    }
    state = {
        'divergence': 0.3,
        'safety_state': 'unsafe'
    }
    
    mitigations = support.generate_mitigations(
        "sensor_spoofing", explanation, state, 0.8
    )
    
    print(f"Generated {len(mitigations)} mitigations:")
    for i, mit in enumerate(mitigations, 1):
        print(f"{i}. [{mit.priority.value.upper()}] {mit.action.value}")
        print(f"   {mit.description}")
