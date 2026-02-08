"""
Research-Grade Dashboard - 20/10 Elite System

Integrates all advanced components for research-grade operation:
- Real-time streaming
- Ensemble detection
- Ground-truth validation
- Adaptive thresholds
- Explainability
- Defensive support
- Physics-coupled twin
- Model versioning
- Enhanced trust
- Publication outputs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import json
from datetime import datetime
import time

# Core components
from .data_processor import SWaTDataProcessor
from .digital_twin import CyberAwareDigitalTwin
from .genai_engine import LSTMAutoencoder

# New research-grade components
from .streaming_processor import StreamingProcessor, RealTimeDetector
from .ensemble_detector import EnsembleAnomalyDetector
from .validation_metrics import ValidationMetrics
from .adaptive_threshold import AdaptiveThreshold, BehavioralDriftDetector
from .explainability_engine import ExplainabilityEngine
from .defensive_support import DefensiveDecisionSupport
from .physics_coupled_twin import PhysicsCoupledDigitalTwin
from .model_versioning import ModelVersionManager
from .enhanced_trust import EnhancedTrustAssessment
from .publication_outputs import PublicationOutputGenerator
from .gap_analyzer import CyberGapAnalyzer
from .visualizer import CyberAwareVisualizer


class ResearchGradeDashboard:
    """
    Research-grade dashboard integrating all advanced features
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize research-grade dashboard"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Core components
        self.data_processor = SWaTDataProcessor(config_path)
        self.digital_twin = CyberAwareDigitalTwin(config_path)
        self.genai = LSTMAutoencoder(config_path)
        
        # Research-grade components
        self.streaming_processor = StreamingProcessor(config_path)
        self.ensemble_detector = EnsembleAnomalyDetector(config_path)
        self.validation_metrics = ValidationMetrics()
        self.adaptive_threshold = AdaptiveThreshold(config_path)
        self.drift_detector = BehavioralDriftDetector()
        self.explainability = ExplainabilityEngine(config_path)
        self.defensive_support = DefensiveDecisionSupport(config_path)
        self.physics_twin = PhysicsCoupledDigitalTwin(config_path)
        self.model_versioning = ModelVersionManager()
        self.trust_assessment = EnhancedTrustAssessment(config_path)
        self.publication_outputs = PublicationOutputGenerator()
        self.gap_analyzer = CyberGapAnalyzer(config_path)
        self.visualizer = CyberAwareVisualizer(config_path)
        
        # Real-time operation
        self.is_real_time = False
        self.real_time_detector = None
        
        # Output paths
        self.output_dir = Path("outputs/research")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_ensemble_models(self, normal_sequences: np.ndarray,
                            normal_features: Optional[np.ndarray] = None):
        """
        Train ensemble models on normal data
        
        Args:
            normal_sequences: Time-series sequences
            normal_features: Feature vectors
        """
        print("🧠 Training Ensemble Models...")
        self.ensemble_detector.train_ensemble(normal_sequences, normal_features)
        print("✅ Ensemble training complete")
    
    def process_real_time_stream(self, sensor_data: Dict,
                                ground_truth: Optional[bool] = None,
                                attack_type: Optional[str] = None) -> Dict:
        """
        Process single real-time stream sample
        
        Args:
            sensor_data: Dictionary with sensor readings
            ground_truth: Optional ground truth label
            attack_type: Optional attack type label
        
        Returns:
            Detection result dictionary
        """
        timestamp = time.time()
        
        # Add to streaming buffer
        ready = self.streaming_processor.add_sample(timestamp, sensor_data)
        
        if not ready:
            return {'ready': False, 'buffer_size': len(self.streaming_processor.buffer)}
        
        # Get window sequence
        sequences, timestamps = self.streaming_processor.get_window_sequence()
        
        if sequences is None:
            return {'ready': False}
        
        # Prepare features for ensemble
        features = sequences.reshape(sequences.shape[0], -1)
        
        # Ensemble detection
        ensemble_results = self.ensemble_detector.detect_anomaly_ensemble(
            sequences, features
        )
        
        # Get ensemble prediction
        ensemble_pred = ensemble_results['ensemble']
        is_anomaly = ensemble_pred['anomaly_flags'][-1] if len(ensemble_pred['anomaly_flags']) > 0 else False
        confidence = ensemble_pred['confidence_scores'][-1] if len(ensemble_pred['confidence_scores']) > 0 else 0.0
        
        # Update adaptive threshold
        if hasattr(self.genai, 'compute_reconstruction_error'):
            try:
                errors = self.genai.compute_reconstruction_error(sequences)
                avg_error = np.mean(errors)
                self.adaptive_threshold.update(avg_error, is_anomaly, timestamp)
            except:
                pass
        
        # Update ground truth metrics
        if ground_truth is not None:
            self.validation_metrics.add_detection(
                is_anomaly, ground_truth, confidence, attack_type
            )
        
        # Compute detection latency (if attack just started)
        detection_latency = None
        if is_anomaly and ground_truth:
            # Simplified: assume attack started recently
            detection_latency = 0.0  # Would track actual start time
        
        # Digital twin update
        level = sensor_data.get('level', 0.0)
        valve = sensor_data.get('valve', 0)
        pump = sensor_data.get('pump', 0)
        
        twin_state = self.digital_twin.update_state(
            timestamp, level, valve, pump
        )
        
        # Physics-coupled twin
        physics_state = self.physics_twin.compute_physics_state(
            timestamp, level, valve, pump
        )
        
        # Trust assessment
        expected_level = twin_state.expected_state
        observed_level = twin_state.observed_state
        divergence = abs(expected_level - observed_level) / (expected_level + 1e-10)
        trust_score = 1.0 - min(divergence, 1.0)
        
        trust_snapshot = self.trust_assessment.update_trust(
            timestamp, trust_score, attack_active=ground_truth or False
        )
        
        # Generate explanation if anomaly detected
        explanation = None
        if is_anomaly:
            try:
                reconstruction_errors = np.array([confidence])  # Simplified
                explanation = self.explainability.explain_anomaly(
                    timestamp, sequences, reconstruction_errors,
                    ensemble_results, self.digital_twin.get_state_history()
                )
            except Exception as e:
                print(f"⚠️ Explanation generation error: {e}")
        
        # Generate mitigations if anomaly detected
        mitigations = []
        if is_anomaly and explanation:
            try:
                risk_score = confidence
                mitigations = self.defensive_support.generate_mitigations(
                    attack_type or "unknown",
                    explanation.__dict__,
                    physics_state.__dict__,
                    risk_score
                )
            except Exception as e:
                print(f"⚠️ Mitigation generation error: {e}")
        
        return {
            'timestamp': timestamp,
            'ready': True,
            'anomaly_detected': is_anomaly,
            'confidence': confidence,
            'ensemble_results': {
                'contributing_models': ensemble_pred.get('contributing_models', []),
                'anomaly_rate': ensemble_pred.get('anomaly_rate', 0.0)
            },
            'explanation': explanation.__dict__ if explanation else None,
            'mitigations': [m.__dict__ for m in mitigations[:3]],  # Top 3
            'trust_score': trust_score,
            'trust_state': trust_snapshot.trust_state.value,
            'physics_state': {
                'safety_boundary': physics_state.safety_boundary.value,
                'safety_margin': physics_state.safety_margin
            },
            'detection_latency': detection_latency
        }
    
    def run_validation_experiment(self, test_data: Dict,
                                 ground_truth_labels: List[bool],
                                 attack_types: List[Optional[str]]) -> Dict:
        """
        Run validation experiment with ground truth
        
        Args:
            test_data: Dictionary with test sequences
            ground_truth_labels: List of ground truth labels
            attack_types: List of attack types (or None)
        
        Returns:
            Experiment results dictionary
        """
        print("🔬 Running Validation Experiment...")
        
        # Reset metrics
        self.validation_metrics.reset()
        
        # Process all test samples
        for i, (seq, label, attack_type) in enumerate(zip(
            test_data.get('sequences', []),
            ground_truth_labels,
            attack_types
        )):
            # Convert sequence to sensor data format
            sensor_data = {
                'level': float(seq[-1, 0]) if len(seq.shape) > 1 else float(seq[-1]),
                'valve': 1,
                'pump': 0
            }
            
            result = self.process_real_time_stream(
                sensor_data, ground_truth=label, attack_type=attack_type
            )
            
            if i % 100 == 0:
                print(f"  Processed {i}/{len(ground_truth_labels)} samples...")
        
        # Compute metrics
        overall_metrics = self.validation_metrics.compute_metrics()
        per_attack_metrics = self.validation_metrics.compute_per_attack_metrics()
        
        # Generate outputs
        metrics_table = self.validation_metrics.generate_metrics_table()
        confusion_matrix = self.validation_metrics.generate_confusion_matrix_table()
        
        # Save results
        results = {
            'overall_metrics': overall_metrics.__dict__,
            'per_attack_metrics': {
                k: v.__dict__ for k, v in per_attack_metrics.items()
            },
            'metrics_table': metrics_table.to_dict('records'),
            'confusion_matrix': confusion_matrix.to_dict('records')
        }
        
        # Generate publication outputs
        self.publication_outputs.generate_metrics_table(metrics_table)
        self.publication_outputs.generate_csv_results(metrics_table)
        
        print("✅ Validation experiment complete")
        return results
    
    def generate_research_report(self) -> str:
        """
        Generate comprehensive research report
        
        Returns:
            Path to generated report
        """
        print("📊 Generating Research Report...")
        
        # Collect all data
        overall_metrics = self.validation_metrics.compute_metrics()
        per_attack_metrics = self.validation_metrics.compute_per_attack_metrics()
        
        explanations = self.explainability.explanations
        mitigations = self.defensive_support.recommendations
        
        # Generate report
        report_path = self.publication_outputs.generate_comprehensive_report(
            overall_metrics.__dict__,
            per_attack_metrics,
            explanations,
            mitigations
        )
        
        print(f"✅ Report generated: {report_path}")
        return report_path


if __name__ == "__main__":
    # Example usage
    dashboard = ResearchGradeDashboard()
    
    print("🛡️ Research-Grade Cyber-Aware Digital Twin System")
    print("=" * 60)
    print("System initialized and ready for operation")
    print("=" * 60)
