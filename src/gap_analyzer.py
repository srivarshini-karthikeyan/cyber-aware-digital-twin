"""
Cyber Gap Analysis Engine (CRITICAL COMPONENT)

For every detected anomaly:
- Determines whether unsafe state occurred
- Measures detection delay
- Identifies why detection failed
- Explicitly identifies gaps (single sensor trust, no validation, etc.)
- Proposes clear, actionable mitigations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from datetime import datetime
import json


class GapCategory(Enum):
    """Categories of cybersecurity gaps"""
    SINGLE_SENSOR_DEPENDENCY = "single_sensor_dependency"
    NO_RATE_VALIDATION = "no_rate_validation"
    ABSOLUTE_THRESHOLD_ONLY = "absolute_threshold_only"
    NO_CROSS_SENSOR_CHECK = "no_cross_sensor_check"
    BLIND_CONTROLLER_TRUST = "blind_controller_trust"
    MISSING_SANITY_CHECK = "missing_sanity_check"
    NO_DIGITAL_TWIN_VALIDATION = "no_digital_twin_validation"
    DELAYED_RESPONSE_ACCEPTANCE = "delayed_response_acceptance"


class SeverityLevel(Enum):
    """Severity levels for gaps and unsafe states"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CybersecurityGap:
    """Represents a cybersecurity gap"""
    gap_id: str
    category: GapCategory
    description: str
    severity: SeverityLevel
    affected_component: str
    detection_delay: float  # seconds
    unsafe_state_occurred: bool
    physical_impact: str
    mitigation: str
    mitigation_priority: int  # 1 = highest


@dataclass
class AttackAnalysis:
    """Complete analysis of an attack scenario"""
    attack_id: str
    attack_type: str
    start_time: float
    end_time: float
    duration: float
    detected: bool
    detection_time: Optional[float]
    detection_delay: Optional[float]
    unsafe_state_occurred: bool
    unsafe_state_start: Optional[float]
    unsafe_state_duration: Optional[float]
    gaps_identified: List[CybersecurityGap]
    root_cause: str
    recommended_mitigations: List[str]
    risk_score: float  # [0, 1]


class CyberGapAnalyzer:
    """
    Analyzes cybersecurity gaps in control system responses to attacks
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize gap analyzer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        gap_config = self.config['gap_analysis']
        self.detection_delay_threshold = gap_config['detection_delay_threshold']
        self.severity_levels = gap_config['unsafe_state_severity_levels']
        self.gap_categories = gap_config['gap_categories']
        
        self.analyses: List[AttackAnalysis] = []
        
    def identify_gaps(self, attack_data: Dict, 
                     digital_twin_states: pd.DataFrame,
                     genai_anomaly_flags: np.ndarray,
                     genai_confidence_scores: np.ndarray) -> List[CybersecurityGap]:
        """
        Identify cybersecurity gaps from attack scenario
        
        Args:
            attack_data: Dictionary with attack information
            digital_twin_states: DataFrame with three-state model history
            genai_anomaly_flags: Boolean array of anomaly detections
            genai_confidence_scores: Confidence scores from GenAI
        
        Returns:
            List of identified gaps
        """
        gaps = []
        
        # Check if attack was detected
        detected = np.any(genai_anomaly_flags)
        detection_time = None
        detection_delay = None
        
        if detected:
            detection_indices = np.where(genai_anomaly_flags)[0]
            detection_time = float(detection_indices[0])
            attack_start = attack_data.get('start_time', 0.0)
            detection_delay = detection_time - attack_start
        else:
            detection_delay = float('inf')
        
        # Check for unsafe states
        unsafe_occurred = False
        unsafe_start = None
        unsafe_duration = None
        
        if 'safety_state' in digital_twin_states.columns:
            unsafe_mask = digital_twin_states['safety_state'].isin(['unsafe', 'critical'])
            if unsafe_mask.any():
                unsafe_occurred = True
                unsafe_indices = digital_twin_states.index[unsafe_mask]
                unsafe_start = float(unsafe_indices[0])
                unsafe_duration = float(len(unsafe_indices))
        
        # Analyze state divergence
        if 'divergence' in digital_twin_states.columns:
            max_divergence = digital_twin_states['divergence'].max()
            avg_divergence = digital_twin_states['divergence'].mean()
        else:
            max_divergence = 0.0
            avg_divergence = 0.0
        
        # Gap 1: Single Sensor Dependency
        if attack_data.get('attack_type') == 'sensor_spoofing':
            gap = CybersecurityGap(
                gap_id=f"gap_001_{attack_data.get('attack_id', 'unknown')}",
                category=GapCategory.SINGLE_SENSOR_DEPENDENCY,
                description="System relies solely on single level sensor without redundancy or cross-validation",
                severity=SeverityLevel.HIGH if unsafe_occurred else SeverityLevel.MEDIUM,
                affected_component="Level Sensor (LIT101)",
                detection_delay=detection_delay if detection_delay != float('inf') else None,
                unsafe_state_occurred=unsafe_occurred,
                physical_impact="Tank overflow or dry run risk" if unsafe_occurred else "Potential safety hazard",
                mitigation="Implement redundant sensors with voting logic or add digital twin-based validation",
                mitigation_priority=1
            )
            gaps.append(gap)
        
        # Gap 2: No Rate of Change Validation
        if max_divergence > 0.1 and detection_delay > self.detection_delay_threshold:
            gap = CybersecurityGap(
                gap_id=f"gap_002_{attack_data.get('attack_id', 'unknown')}",
                category=GapCategory.NO_RATE_VALIDATION,
                description="System does not validate rate of change, allowing sudden unrealistic transitions",
                severity=SeverityLevel.MEDIUM,
                affected_component="Control Logic",
                detection_delay=detection_delay if detection_delay != float('inf') else None,
                unsafe_state_occurred=unsafe_occurred,
                physical_impact="Undetected rapid state changes leading to unsafe conditions",
                mitigation="Add rate-of-change monitoring with configurable thresholds",
                mitigation_priority=2
            )
            gaps.append(gap)
        
        # Gap 3: Absolute Threshold Only
        if unsafe_occurred and not detected:
            gap = CybersecurityGap(
                gap_id=f"gap_003_{attack_data.get('attack_id', 'unknown')}",
                category=GapCategory.ABSOLUTE_THRESHOLD_ONLY,
                description="System relies only on absolute thresholds without predictive or trend-based warnings",
                severity=SeverityLevel.HIGH,
                affected_component="Safety System",
                detection_delay=detection_delay if detection_delay != float('inf') else None,
                unsafe_state_occurred=True,
                physical_impact="Unsafe state reached before alarm triggers",
                mitigation="Implement predictive thresholds and early warning systems based on trend analysis",
                mitigation_priority=1
            )
            gaps.append(gap)
        
        # Gap 4: No Cross-Sensor Check
        if attack_data.get('attack_type') in ['sensor_spoofing', 'frozen_sensor']:
            gap = CybersecurityGap(
                gap_id=f"gap_004_{attack_data.get('attack_id', 'unknown')}",
                category=GapCategory.NO_CROSS_SENSOR_CHECK,
                description="No validation between sensor readings and expected behavior from other sensors/actuators",
                severity=SeverityLevel.MEDIUM,
                affected_component="Sensor Validation Logic",
                detection_delay=detection_delay if detection_delay != float('inf') else None,
                unsafe_state_occurred=unsafe_occurred,
                physical_impact="Inconsistent sensor data not caught by system",
                mitigation="Implement cross-sensor consistency checks using digital twin predictions",
                mitigation_priority=2
            )
            gaps.append(gap)
        
        # Gap 5: Blind Controller Trust
        if max_divergence > 0.2:
            gap = CybersecurityGap(
                gap_id=f"gap_005_{attack_data.get('attack_id', 'unknown')}",
                category=GapCategory.BLIND_CONTROLLER_TRUST,
                description="Controller blindly trusts sensor readings without independent verification",
                severity=SeverityLevel.HIGH,
                affected_component="Controller Logic",
                detection_delay=detection_delay if detection_delay != float('inf') else None,
                unsafe_state_occurred=unsafe_occurred,
                physical_impact="Controller makes decisions based on compromised sensor data",
                mitigation="Implement digital twin as independent verification layer before control actions",
                mitigation_priority=1
            )
            gaps.append(gap)
        
        # Gap 6: Missing Sanity Check
        if detection_delay == float('inf') or detection_delay > 10.0:
            gap = CybersecurityGap(
                gap_id=f"gap_006_{attack_data.get('attack_id', 'unknown')}",
                category=GapCategory.MISSING_SANITY_CHECK,
                description="No sanity checks to detect physically impossible sensor readings",
                severity=SeverityLevel.MEDIUM,
                affected_component="Input Validation",
                detection_delay=detection_delay if detection_delay != float('inf') else None,
                unsafe_state_occurred=unsafe_occurred,
                physical_impact="Impossible sensor values accepted without question",
                mitigation="Add physics-based sanity checks (e.g., level cannot decrease when valve is closed and pump is off)",
                mitigation_priority=2
            )
            gaps.append(gap)
        
        # Gap 7: No Digital Twin Validation
        if max_divergence > 0.15 and not detected:
            gap = CybersecurityGap(
                gap_id=f"gap_007_{attack_data.get('attack_id', 'unknown')}",
                category=GapCategory.NO_DIGITAL_TWIN_VALIDATION,
                description="System does not use digital twin predictions to validate sensor readings",
                severity=SeverityLevel.HIGH,
                affected_component="Validation Layer",
                detection_delay=detection_delay if detection_delay != float('inf') else None,
                unsafe_state_occurred=unsafe_occurred,
                physical_impact="Large divergence between expected and observed states goes undetected",
                mitigation="Integrate digital twin as real-time validation layer with automatic divergence alerts",
                mitigation_priority=1
            )
            gaps.append(gap)
        
        return gaps
    
    def analyze_attack(self, attack_data: Dict,
                      digital_twin_states: pd.DataFrame,
                      genai_anomaly_flags: np.ndarray,
                      genai_confidence_scores: np.ndarray) -> AttackAnalysis:
        """
        Complete attack analysis
        
        Returns:
            AttackAnalysis with all findings
        """
        # Identify gaps
        gaps = self.identify_gaps(
            attack_data, digital_twin_states,
            genai_anomaly_flags, genai_confidence_scores
        )
        
        # Detection analysis
        detected = np.any(genai_anomaly_flags)
        detection_time = None
        detection_delay = None
        
        if detected:
            detection_indices = np.where(genai_anomaly_flags)[0]
            detection_time = float(detection_indices[0])
            attack_start = attack_data.get('start_time', 0.0)
            detection_delay = detection_time - attack_start
        
        # Unsafe state analysis
        unsafe_occurred = False
        unsafe_start = None
        unsafe_duration = None
        
        if 'safety_state' in digital_twin_states.columns:
            unsafe_mask = digital_twin_states['safety_state'].isin(['unsafe', 'critical'])
            if unsafe_mask.any():
                unsafe_occurred = True
                unsafe_indices = digital_twin_states.index[unsafe_mask]
                unsafe_start = float(unsafe_indices[0])
                unsafe_duration = float(len(unsafe_indices))
        
        # Root cause
        if gaps:
            primary_gap = max(gaps, key=lambda g: g.mitigation_priority)
            root_cause = primary_gap.description
        else:
            root_cause = "Unknown - requires further investigation"
        
        # Recommended mitigations
        mitigations = []
        for gap in sorted(gaps, key=lambda g: g.mitigation_priority):
            if gap.mitigation not in mitigations:
                mitigations.append(gap.mitigation)
        
        # Risk score
        risk_score = self.compute_risk_score(
            detected, detection_delay, unsafe_occurred,
            unsafe_duration, len(gaps)
        )
        
        # Create analysis
        analysis = AttackAnalysis(
            attack_id=attack_data.get('attack_id', f"attack_{len(self.analyses)}"),
            attack_type=attack_data.get('attack_type', 'unknown'),
            start_time=attack_data.get('start_time', 0.0),
            end_time=attack_data.get('end_time', 0.0),
            duration=attack_data.get('duration', 0.0),
            detected=detected,
            detection_time=detection_time,
            detection_delay=detection_delay,
            unsafe_state_occurred=unsafe_occurred,
            unsafe_state_start=unsafe_start,
            unsafe_state_duration=unsafe_duration,
            gaps_identified=gaps,
            root_cause=root_cause,
            recommended_mitigations=mitigations,
            risk_score=risk_score
        )
        
        self.analyses.append(analysis)
        return analysis
    
    def compute_risk_score(self, detected: bool, detection_delay: Optional[float],
                          unsafe_occurred: bool, unsafe_duration: Optional[float],
                          num_gaps: int) -> float:
        """
        Compute overall risk score [0, 1]
        """
        score = 0.0
        
        # Detection component
        if not detected:
            score += 0.4
        elif detection_delay and detection_delay > self.detection_delay_threshold:
            score += 0.2 * min(detection_delay / 30.0, 1.0)
        
        # Unsafe state component
        if unsafe_occurred:
            score += 0.4
            if unsafe_duration:
                score += 0.1 * min(unsafe_duration / 60.0, 1.0)
        
        # Gap count component
        score += 0.1 * min(num_gaps / 5.0, 1.0)
        
        return min(score, 1.0)
    
    def generate_incident_report(self, analysis: AttackAnalysis) -> Dict:
        """
        Generate human-readable incident report
        """
        report = {
            'incident_id': analysis.attack_id,
            'timestamp': datetime.now().isoformat(),
            'attack_type': analysis.attack_type,
            'summary': {
                'detected': analysis.detected,
                'detection_delay_seconds': analysis.detection_delay,
                'unsafe_state_occurred': analysis.unsafe_state_occurred,
                'unsafe_state_duration_seconds': analysis.unsafe_state_duration,
                'risk_score': analysis.risk_score
            },
            'root_cause': analysis.root_cause,
            'gaps_identified': [
                {
                    'category': gap.category.value,
                    'description': gap.description,
                    'severity': gap.severity.value,
                    'affected_component': gap.affected_component
                }
                for gap in analysis.gaps_identified
            ],
            'recommended_mitigations': analysis.recommended_mitigations,
            'physical_impact': [
                gap.physical_impact for gap in analysis.gaps_identified
                if gap.unsafe_state_occurred
            ]
        }
        
        return report
    
    def export_analysis(self, filepath: str):
        """Export all analyses to JSON"""
        export_data = []
        for analysis in self.analyses:
            report = self.generate_incident_report(analysis)
            export_data.append(report)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_gap_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all gaps"""
        all_gaps = []
        for analysis in self.analyses:
            for gap in analysis.gaps_identified:
                all_gaps.append({
                    'Attack ID': analysis.attack_id,
                    'Attack Type': analysis.attack_type,
                    'Gap Category': gap.category.value,
                    'Severity': gap.severity.value,
                    'Affected Component': gap.affected_component,
                    'Unsafe State': gap.unsafe_state_occurred,
                    'Detection Delay (s)': gap.detection_delay,
                    'Mitigation Priority': gap.mitigation_priority
                })
        
        return pd.DataFrame(all_gaps)


if __name__ == "__main__":
    # Example usage
    analyzer = CyberGapAnalyzer()
    
    # Dummy attack data
    attack_data = {
        'attack_id': 'test_001',
        'attack_type': 'sensor_spoofing',
        'start_time': 10.0,
        'end_time': 50.0,
        'duration': 40.0
    }
    
    # Dummy states
    states = pd.DataFrame({
        'divergence': [0.0] * 10 + [0.3] * 40 + [0.0] * 10,
        'safety_state': ['safe'] * 10 + ['unsafe'] * 40 + ['safe'] * 10
    })
    
    # Dummy GenAI outputs
    anomaly_flags = np.array([False] * 20 + [True] * 40 + [False] * 10)
    confidence_scores = np.array([0.0] * 20 + [0.8] * 40 + [0.0] * 10)
    
    # Analyze
    analysis = analyzer.analyze_attack(
        attack_data, states, anomaly_flags, confidence_scores
    )
    
    print(f"Attack detected: {analysis.detected}")
    print(f"Detection delay: {analysis.detection_delay} seconds")
    print(f"Gaps identified: {len(analysis.gaps_identified)}")
    print(f"Risk score: {analysis.risk_score:.2f}")
