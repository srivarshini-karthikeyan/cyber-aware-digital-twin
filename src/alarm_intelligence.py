"""
Alarm Intelligence Layer
Elite Enhancement for Production IDS

Post-processes ML detections to:
- Reduce false alarms through temporal correlation
- Classify severity
- Aggregate related alarms
- Apply contextual filtering
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json


class AlarmSeverity(Enum):
    """Alarm severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alarm:
    """Structured alarm representation"""
    alarm_id: str
    timestamp: float
    sensor_id: str
    attack_type: Optional[str]
    severity: AlarmSeverity
    confidence: float
    anomaly_score: float
    safety_impact: float
    explanation: Dict
    mitigation: List[str]
    is_suppressed: bool = False
    suppression_reason: Optional[str] = None


class AlarmIntelligence:
    """
    Intelligent alarm processing to reduce false positives
    """
    
    def __init__(self,
                 temporal_window: int = 3,  # Require N consecutive anomalies
                 suppression_window: float = 60.0,  # Seconds to suppress duplicates
                 severity_thresholds: Optional[Dict] = None):
        """
        Initialize alarm intelligence
        
        Args:
            temporal_window: Number of consecutive samples required for alarm
            suppression_window: Time window to suppress duplicate alarms (seconds)
            severity_thresholds: Custom severity thresholds
        """
        self.temporal_window = temporal_window
        self.suppression_window = suppression_window
        
        # Default severity thresholds
        self.severity_thresholds = severity_thresholds or {
            'critical': {'score': 0.9, 'safety': 0.8},
            'high': {'score': 0.75, 'safety': 0.5},
            'medium': {'score': 0.6, 'safety': 0.3},
            'low': {'score': 0.0, 'safety': 0.0}
        }
        
        # Alarm history
        self.recent_alarms: List[Alarm] = []
        self.anomaly_buffer: List[Dict] = []  # Buffer for temporal correlation
        self.alarm_counter = 0
    
    def classify_severity(self,
                         anomaly_score: float,
                         attack_type: Optional[str],
                         safety_impact: float) -> AlarmSeverity:
        """
        Classify alarm severity
        
        Args:
            anomaly_score: ML anomaly score [0, 1]
            attack_type: Detected attack type
            safety_impact: Safety impact score [0, 1]
        
        Returns:
            AlarmSeverity level
        """
        # Critical: High score + high safety impact
        if (anomaly_score >= self.severity_thresholds['critical']['score'] and
            safety_impact >= self.severity_thresholds['critical']['safety']):
            return AlarmSeverity.CRITICAL
        
        # High: High score OR high safety impact
        if (anomaly_score >= self.severity_thresholds['high']['score'] or
            safety_impact >= self.severity_thresholds['high']['safety']):
            return AlarmSeverity.HIGH
        
        # Medium: Moderate score
        if anomaly_score >= self.severity_thresholds['medium']['score']:
            return AlarmSeverity.MEDIUM
        
        # Low: Everything else
        return AlarmSeverity.LOW
    
    def check_temporal_correlation(self, current_anomaly: Dict) -> bool:
        """
        Check if anomaly is part of sustained pattern (not isolated spike)
        
        Args:
            current_anomaly: Current anomaly detection
        
        Returns:
            True if anomaly is sustained (should trigger alarm)
        """
        self.anomaly_buffer.append(current_anomaly)
        
        # Keep only recent buffer
        current_time = current_anomaly.get('timestamp', datetime.now().timestamp())
        self.anomaly_buffer = [
            a for a in self.anomaly_buffer
            if current_time - a.get('timestamp', 0) < self.suppression_window
        ]
        
        # Check if we have sustained anomalies
        if len(self.anomaly_buffer) >= self.temporal_window:
            # Check if recent anomalies are consistent
            recent_scores = [a.get('anomaly_score', 0) for a in self.anomaly_buffer[-self.temporal_window:]]
            avg_score = np.mean(recent_scores)
            
            # If average score is above threshold, it's sustained
            return avg_score > 0.5
        
        return False
    
    def check_duplicate_suppression(self, alarm: Alarm) -> bool:
        """
        Check if alarm should be suppressed as duplicate
        
        Args:
            alarm: Alarm to check
        
        Returns:
            True if alarm should be suppressed
        """
        current_time = alarm.timestamp
        
        # Check for similar recent alarms
        for recent_alarm in self.recent_alarms:
            time_diff = current_time - recent_alarm.timestamp
            
            if time_diff > self.suppression_window:
                continue  # Too old
            
            # Check if same sensor and similar type
            if (recent_alarm.sensor_id == alarm.sensor_id and
                recent_alarm.attack_type == alarm.attack_type):
                return True  # Duplicate
        
        return False
    
    def check_contextual_filtering(self, alarm: Alarm, context: Dict) -> bool:
        """
        Check if alarm should be filtered based on context
        
        Args:
            alarm: Alarm to filter
            context: Operational context
        
        Returns:
            True if alarm should be suppressed
        """
        # Maintenance mode: suppress low/medium alarms
        if context.get('maintenance_mode', False):
            if alarm.severity in [AlarmSeverity.LOW, AlarmSeverity.MEDIUM]:
                return True
        
        # Startup phase: suppress low alarms
        if context.get('startup_phase', False):
            if alarm.severity == AlarmSeverity.LOW:
                return True
        
        # Known events: suppress if matches
        known_events = context.get('known_events', [])
        for event in known_events:
            if (event.get('sensor_id') == alarm.sensor_id and
                abs(event.get('timestamp', 0) - alarm.timestamp) < 60):
                return True
        
        return False
    
    def process_detection(self,
                         detection: Dict,
                         context: Optional[Dict] = None) -> Optional[Alarm]:
        """
        Process ML detection through alarm intelligence
        
        Args:
            detection: ML detection result
            context: Operational context
        
        Returns:
            Alarm object if alarm should be raised, None otherwise
        """
        context = context or {}
        
        # Extract detection info
        sensor_id = detection.get('sensor_id', 'unknown')
        anomaly_score = detection.get('anomaly_score', 0.0)
        attack_type = detection.get('attack_type')
        safety_impact = detection.get('safety_impact', 0.0)
        timestamp = detection.get('timestamp', datetime.now().timestamp())
        explanation = detection.get('explanation', {})
        mitigation = detection.get('mitigation', [])
        
        # Temporal correlation: require sustained anomalies
        if not self.check_temporal_correlation({
            'timestamp': timestamp,
            'anomaly_score': anomaly_score,
            'sensor_id': sensor_id
        }):
            return None  # Isolated spike, suppress
        
        # Classify severity
        severity = self.classify_severity(anomaly_score, attack_type, safety_impact)
        
        # Create alarm
        self.alarm_counter += 1
        alarm = Alarm(
            alarm_id=f"ALARM_{self.alarm_counter:06d}",
            timestamp=timestamp,
            sensor_id=sensor_id,
            attack_type=attack_type,
            severity=severity,
            confidence=anomaly_score,
            anomaly_score=anomaly_score,
            safety_impact=safety_impact,
            explanation=explanation,
            mitigation=mitigation
        )
        
        # Check duplicate suppression
        if self.check_duplicate_suppression(alarm):
            alarm.is_suppressed = True
            alarm.suppression_reason = "Duplicate alarm (recent similar alarm)"
            return alarm  # Return but mark as suppressed
        
        # Check contextual filtering
        if self.check_contextual_filtering(alarm, context):
            alarm.is_suppressed = True
            alarm.suppression_reason = "Contextual filtering (maintenance/startup)"
            return alarm
        
        # Add to history
        self.recent_alarms.append(alarm)
        
        # Keep only recent alarms
        current_time = timestamp
        self.recent_alarms = [
            a for a in self.recent_alarms
            if current_time - a.timestamp < self.suppression_window * 2
        ]
        
        return alarm
    
    def get_alarm_statistics(self) -> Dict:
        """Get alarm processing statistics"""
        total_detections = len(self.anomaly_buffer)
        total_alarms = len([a for a in self.recent_alarms if not a.is_suppressed])
        suppressed_alarms = len([a for a in self.recent_alarms if a.is_suppressed])
        
        severity_counts = {
            'critical': len([a for a in self.recent_alarms if a.severity == AlarmSeverity.CRITICAL and not a.is_suppressed]),
            'high': len([a for a in self.recent_alarms if a.severity == AlarmSeverity.HIGH and not a.is_suppressed]),
            'medium': len([a for a in self.recent_alarms if a.severity == AlarmSeverity.MEDIUM and not a.is_suppressed]),
            'low': len([a for a in self.recent_alarms if a.severity == AlarmSeverity.LOW and not a.is_suppressed])
        }
        
        suppression_rate = suppressed_alarms / len(self.recent_alarms) if len(self.recent_alarms) > 0 else 0.0
        
        return {
            'total_detections': total_detections,
            'total_alarms': total_alarms,
            'suppressed_alarms': suppressed_alarms,
            'suppression_rate': suppression_rate,
            'severity_distribution': severity_counts,
            'alarm_reduction_factor': total_detections / total_alarms if total_alarms > 0 else 1.0
        }


if __name__ == "__main__":
    # Test alarm intelligence
    intelligence = AlarmIntelligence(temporal_window=3)
    
    # Simulate sustained anomaly (should trigger alarm)
    print("Testing sustained anomaly...")
    for i in range(5):
        detection = {
            'sensor_id': 'sensor_1',
            'anomaly_score': 0.8,
            'attack_type': 'sensor_spoofing',
            'safety_impact': 0.7,
            'timestamp': datetime.now().timestamp() + i,
            'explanation': {},
            'mitigation': []
        }
        alarm = intelligence.process_detection(detection)
        if alarm:
            print(f"  Alarm {i+1}: {alarm.severity.value}, Suppressed: {alarm.is_suppressed}")
    
    # Simulate isolated spike (should be suppressed)
    print("\nTesting isolated spike...")
    detection = {
        'sensor_id': 'sensor_2',
        'anomaly_score': 0.9,
        'attack_type': None,
        'safety_impact': 0.1,
        'timestamp': datetime.now().timestamp() + 10,
        'explanation': {},
        'mitigation': []
    }
    alarm = intelligence.process_detection(detection)
    if alarm:
        print(f"  Alarm: {alarm.severity.value}, Suppressed: {alarm.is_suppressed}")
    else:
        print("  Alarm suppressed (isolated spike)")
    
    stats = intelligence.get_alarm_statistics()
    print(f"\nStatistics: {stats}")
