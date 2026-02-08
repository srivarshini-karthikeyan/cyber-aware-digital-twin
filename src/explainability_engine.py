"""
Explainability & Transparency Engine

Provides human-interpretable explanations for every detected anomaly
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import yaml


@dataclass
class AnomalyExplanation:
    """Structured explanation for an anomaly"""
    timestamp: float
    anomaly_confidence: float
    primary_cause: str
    sensor_contributions: Dict[str, float]
    feature_attributions: Dict[str, float]
    temporal_evolution: str
    human_readable: str
    recommended_action: str


class ExplainabilityEngine:
    """
    Generates explainable interpretations of anomalies
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize explainability engine"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_names = self.config['data']['features']
        self.explanations: List[AnomalyExplanation] = []
    
    def explain_anomaly(self, timestamp: float,
                        sequences: np.ndarray,
                        reconstruction_errors: np.ndarray,
                        ensemble_results: Dict,
                        digital_twin_states: Optional[pd.DataFrame] = None) -> AnomalyExplanation:
        """
        Generate comprehensive explanation for detected anomaly
        
        Args:
            timestamp: Detection timestamp
            sequences: Input sequences
            reconstruction_errors: Reconstruction errors per feature
            ensemble_results: Results from ensemble detector
            digital_twin_states: Digital twin state history
        
        Returns:
            AnomalyExplanation object
        """
        # Compute sensor-level contributions
        sensor_contributions = self._compute_sensor_contributions(
            sequences, reconstruction_errors
        )
        
        # Compute feature attributions
        feature_attributions = self._compute_feature_attributions(
            sequences, reconstruction_errors
        )
        
        # Identify primary cause
        primary_cause = self._identify_primary_cause(
            sensor_contributions, feature_attributions, ensemble_results
        )
        
        # Analyze temporal evolution
        temporal_evolution = self._analyze_temporal_evolution(
            sequences, digital_twin_states
        )
        
        # Generate human-readable explanation
        human_readable = self._generate_human_readable(
            primary_cause, sensor_contributions, temporal_evolution
        )
        
        # Recommend action
        recommended_action = self._recommend_action(
            primary_cause, sensor_contributions
        )
        
        # Get ensemble confidence
        anomaly_confidence = ensemble_results.get('ensemble', {}).get(
            'confidence_scores', [0.0]
        )[-1] if ensemble_results else 0.0
        
        explanation = AnomalyExplanation(
            timestamp=timestamp,
            anomaly_confidence=anomaly_confidence,
            primary_cause=primary_cause,
            sensor_contributions=sensor_contributions,
            feature_attributions=feature_attributions,
            temporal_evolution=temporal_evolution,
            human_readable=human_readable,
            recommended_action=recommended_action
        )
        
        self.explanations.append(explanation)
        return explanation
    
    def _compute_sensor_contributions(self, sequences: np.ndarray,
                                     reconstruction_errors: np.ndarray) -> Dict[str, float]:
        """
        Compute contribution of each sensor to anomaly
        
        Returns:
            Dictionary mapping sensor name to contribution score
        """
        if len(reconstruction_errors.shape) == 1:
            # Per-sequence error, compute per-feature
            errors_per_feature = np.mean(
                np.abs(sequences[-1] - sequences[-1].mean(axis=0, keepdims=True)),
                axis=0
            )
        else:
            errors_per_feature = np.mean(reconstruction_errors, axis=0)
        
        # Normalize to [0, 1]
        total_error = np.sum(errors_per_feature)
        if total_error > 0:
            contributions = errors_per_feature / total_error
        else:
            contributions = np.ones_like(errors_per_feature) / len(errors_per_feature)
        
        # Map to sensor names
        sensor_contributions = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(contributions):
                sensor_contributions[feature_name] = float(contributions[i])
        
        return sensor_contributions
    
    def _compute_feature_attributions(self, sequences: np.ndarray,
                                     reconstruction_errors: np.ndarray) -> Dict[str, float]:
        """
        Compute feature-wise anomaly attribution
        
        Returns:
            Dictionary mapping feature to attribution score
        """
        # Use gradient-like approach: how much does each feature deviate?
        last_sequence = sequences[-1] if len(sequences.shape) > 2 else sequences
        
        # Compute deviation from expected pattern
        if len(last_sequence.shape) == 2:
            # Time-series: compute deviation over time
            deviations = np.std(last_sequence, axis=0)
        else:
            deviations = np.abs(last_sequence - np.mean(last_sequence))
        
        # Normalize
        total_deviation = np.sum(deviations)
        if total_deviation > 0:
            attributions = deviations / total_deviation
        else:
            attributions = np.ones_like(deviations) / len(deviations)
        
        feature_attributions = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(attributions):
                feature_attributions[feature_name] = float(attributions[i])
        
        return feature_attributions
    
    def _identify_primary_cause(self, sensor_contributions: Dict,
                               feature_attributions: Dict,
                               ensemble_results: Dict) -> str:
        """
        Identify primary cause of anomaly
        
        Returns:
            Human-readable cause description
        """
        # Find sensor with highest contribution
        max_sensor = max(sensor_contributions.items(), key=lambda x: x[1])
        max_feature = max(feature_attributions.items(), key=lambda x: x[1])
        
        # Check ensemble model contributions
        contributing_models = ensemble_results.get('ensemble', {}).get(
            'contributing_models', []
        )
        
        if max_sensor[1] > 0.5:
            sensor_name = max_sensor[0]
            if sensor_name == 'LIT101':
                return f"Level sensor ({sensor_name}) shows significant deviation - possible spoofing or sensor failure"
            elif sensor_name == 'MV101':
                return f"Inlet valve ({sensor_name}) behavior inconsistent - possible manipulation"
            elif sensor_name == 'P101':
                return f"Outlet pump ({sensor_name}) shows anomalous pattern - possible control attack"
            else:
                return f"Sensor {sensor_name} is primary contributor to anomaly"
        elif len(contributing_models) > 0:
            return f"Multiple models detected anomaly - ensemble confidence from {', '.join(contributing_models)}"
        else:
            return "Anomaly detected across multiple sensors - possible coordinated attack"
    
    def _analyze_temporal_evolution(self, sequences: np.ndarray,
                                   digital_twin_states: Optional[pd.DataFrame]) -> str:
        """
        Analyze temporal evolution of anomaly
        
        Returns:
            Description of temporal pattern
        """
        if sequences is None or len(sequences) == 0:
            return "Insufficient data for temporal analysis"
        
        # Analyze trend in last sequence
        if len(sequences.shape) == 3:
            last_seq = sequences[-1]  # Shape: (sequence_length, n_features)
        else:
            last_seq = sequences
        
        # Compute rate of change
        if len(last_seq.shape) == 2 and last_seq.shape[0] > 1:
            rates = np.diff(last_seq, axis=0)
            avg_rate = np.mean(np.abs(rates))
            
            if avg_rate > 0.1:
                return f"Rapid change detected (rate: {avg_rate:.3f}) - anomaly is escalating"
            elif avg_rate < 0.01:
                return "Stable but anomalous pattern - possible frozen sensor or replay attack"
            else:
                return f"Gradual anomaly evolution (rate: {avg_rate:.3f}) - possible slow manipulation"
        else:
            return "Anomaly detected at current time step"
    
    def _generate_human_readable(self, primary_cause: str,
                                sensor_contributions: Dict,
                                temporal_evolution: str) -> str:
        """
        Generate human-readable explanation
        
        Returns:
            Natural language explanation
        """
        explanation = f"ANOMALY DETECTED\n\n"
        explanation += f"Primary Cause: {primary_cause}\n\n"
        explanation += f"Temporal Pattern: {temporal_evolution}\n\n"
        explanation += "Sensor Contributions:\n"
        
        for sensor, contribution in sorted(sensor_contributions.items(), 
                                         key=lambda x: x[1], reverse=True):
            explanation += f"  - {sensor}: {contribution:.1%} contribution\n"
        
        return explanation
    
    def _recommend_action(self, primary_cause: str,
                         sensor_contributions: Dict) -> str:
        """
        Recommend action based on anomaly
        
        Returns:
            Recommended action string
        """
        max_sensor = max(sensor_contributions.items(), key=lambda x: x[1])
        
        if 'LIT101' in primary_cause or max_sensor[0] == 'LIT101':
            return "IMMEDIATE: Validate level sensor reading with redundant sensor or digital twin prediction. Consider fail-safe valve closure if divergence exceeds safety threshold."
        elif 'MV101' in primary_cause or max_sensor[0] == 'MV101':
            return "IMMEDIATE: Check valve actuator status. Verify valve position matches control command. Enable rate-of-change monitoring."
        elif 'P101' in primary_cause or max_sensor[0] == 'P101':
            return "IMMEDIATE: Verify pump status and flow rate. Check for control command manipulation. Enable cross-sensor validation."
        else:
            return "IMMEDIATE: Investigate multi-sensor anomaly. Enable enhanced monitoring. Review recent control commands for unauthorized changes."
    
    def generate_explanation_report(self) -> pd.DataFrame:
        """
        Generate explanation report table
        
        Returns:
            DataFrame with explanations
        """
        data = []
        for exp in self.explanations:
            data.append({
                'Timestamp': exp.timestamp,
                'Confidence': f"{exp.anomaly_confidence:.2%}",
                'Primary Cause': exp.primary_cause[:50] + "..." if len(exp.primary_cause) > 50 else exp.primary_cause,
                'Top Sensor': max(exp.sensor_contributions.items(), key=lambda x: x[1])[0],
                'Recommended Action': exp.recommended_action[:50] + "..." if len(exp.recommended_action) > 50 else exp.recommended_action
            })
        
        return pd.DataFrame(data)
    
    def get_sensor_importance_plot_data(self) -> Dict:
        """
        Get data for sensor importance visualization
        
        Returns:
            Dictionary with sensor importance data
        """
        if len(self.explanations) == 0:
            return {}
        
        # Aggregate sensor contributions across all explanations
        sensor_totals = {}
        for exp in self.explanations:
            for sensor, contribution in exp.sensor_contributions.items():
                if sensor not in sensor_totals:
                    sensor_totals[sensor] = []
                sensor_totals[sensor].append(contribution)
        
        # Compute averages
        sensor_importance = {
            sensor: np.mean(contributions)
            for sensor, contributions in sensor_totals.items()
        }
        
        return sensor_importance


if __name__ == "__main__":
    # Example usage
    explainer = ExplainabilityEngine()
    
    # Dummy data
    sequences = np.random.rand(1, 60, 3)
    reconstruction_errors = np.array([0.3, 0.1, 0.05])
    ensemble_results = {
        'ensemble': {
            'confidence_scores': np.array([0.8]),
            'contributing_models': ['lstm', 'isolation']
        }
    }
    
    # Generate explanation
    explanation = explainer.explain_anomaly(
        100.0, sequences, reconstruction_errors, ensemble_results
    )
    
    print(explanation.human_readable)
    print(f"\nRecommended Action: {explanation.recommended_action}")
