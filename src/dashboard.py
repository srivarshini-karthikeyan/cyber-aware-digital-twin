"""
Main Dashboard - Orchestrates entire system

This is the main entry point that:
1. Loads and processes SWaT data
2. Trains GenAI model
3. Generates attack scenarios
4. Runs digital twin simulation
5. Performs gap analysis
6. Creates visualizations
7. Generates reports
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
from typing import Dict, List, Optional
import argparse

from .data_processor import SWaTDataProcessor
from .digital_twin import CyberAwareDigitalTwin
from .genai_engine import LSTMAutoencoder
from .gap_analyzer import CyberGapAnalyzer
from .attack_generator import AttackGenerator
from .visualizer import CyberAwareVisualizer
from .advanced_features import (
    TrustDegradationIndex, AttackLatencyAnalyzer,
    SilentFailureDetector, MitigationSimulator
)


class CyberAwareDashboard:
    """
    Main dashboard orchestrating all components
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize dashboard"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_processor = SWaTDataProcessor(config_path)
        self.digital_twin = CyberAwareDigitalTwin(config_path)
        self.genai = LSTMAutoencoder(config_path)
        self.gap_analyzer = CyberGapAnalyzer(config_path)
        self.attack_generator = AttackGenerator(config_path)
        self.visualizer = CyberAwareVisualizer(config_path)
        
        # Advanced features
        self.tdi = TrustDegradationIndex(config_path)
        self.latency_analyzer = AttackLatencyAnalyzer(config_path)
        self.silent_failure_detector = SilentFailureDetector(config_path)
        self.mitigation_simulator = MitigationSimulator(config_path)
        
        # Output paths
        self.output_path = Path(self.config['visualization']['output_path'])
        self.report_path = Path(self.config['visualization']['report_path'])
        self.results_path = Path("outputs/results")
        
        # Create directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def process_data(self, data_file: str) -> Dict:
        """
        Process SWaT dataset
        
        Returns:
            Processed data dictionary
        """
        print("📊 Processing SWaT dataset...")
        data = self.data_processor.prepare_training_data(data_file)
        print(f"✅ Processed {len(data['sequences'])} sequences")
        print(f"   Normal samples: {data['metadata']['normal_samples']}")
        print(f"   Attack samples: {data['metadata']['attack_samples']}")
        return data
    
    def train_genai_model(self, train_sequences: np.ndarray,
                         val_sequences: Optional[np.ndarray] = None,
                         save_model: bool = True) -> Dict:
        """
        Train GenAI model
        
        Returns:
            Training history
        """
        print("🧠 Training GenAI model (LSTM Autoencoder)...")
        
        # Build model
        input_shape = (train_sequences.shape[1], train_sequences.shape[2])
        self.genai.build_model(input_shape)
        
        # Train
        history = self.genai.train(train_sequences, val_sequences, verbose=1)
        
        # Save model
        if save_model:
            model_path = "models/saved_models/genai_autoencoder.h5"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.genai.save_model(model_path)
            print(f"✅ Model saved to {model_path}")
        
        print("✅ GenAI model trained successfully")
        return history
    
    def run_attack_scenario(self, attack_type: str = "sensor_spoofing") -> Dict:
        """
        Run complete attack scenario analysis
        
        Returns:
            Complete analysis results
        """
        print(f"🎯 Running attack scenario: {attack_type}")
        
        # Generate attack
        if attack_type == "sensor_spoofing":
            scenario = self.attack_generator.generate_sensor_spoofing_attack(duration=60)
        elif attack_type == "slow_manipulation":
            scenario = self.attack_generator.generate_slow_manipulation_attack(duration=120)
        elif attack_type == "frozen_sensor":
            scenario = self.attack_generator.generate_frozen_sensor_attack(duration=60)
        elif attack_type == "delayed_response":
            scenario = self.attack_generator.generate_delayed_response_attack(duration=60)
        else:
            scenario = self.attack_generator.generate_sensor_spoofing_attack(duration=60)
        
        # Get digital twin states
        states_df = scenario['digital_twin_history']
        
        # Convert states to sequences for GenAI
        # (In practice, would use actual sensor data)
        sequences = self._states_to_sequences(states_df)
        
        # Run GenAI detection
        anomaly_flags, confidence_scores = self.genai.detect_anomalies(sequences)
        
        # Perform gap analysis
        analysis = self.gap_analyzer.analyze_attack(
            scenario, states_df, anomaly_flags, confidence_scores
        )
        
        # Compute Trust Degradation Index
        tdi_data = self._compute_tdi(states_df)
        
        # Analyze latency
        latency = self.latency_analyzer.analyze_latency(
            analysis.attack_id,
            analysis.start_time,
            analysis.detection_time,
            analysis.unsafe_state_start
        )
        
        # Create visualizations
        gap_summary = self.gap_analyzer.get_gap_summary_table()
        saved_plots = self.visualizer.create_comprehensive_dashboard(
            states_df, analysis.__dict__, gap_summary, tdi_data,
            output_prefix=f"{scenario['attack_id']}"
        )
        
        # Generate incident report
        incident_report = self.gap_analyzer.generate_incident_report(analysis)
        
        # Save results
        results = {
            'scenario': scenario,
            'analysis': analysis.__dict__,
            'latency': latency.__dict__,
            'tdi_data': tdi_data.to_dict('records') if not tdi_data.empty else [],
            'incident_report': incident_report,
            'plots': saved_plots
        }
        
        results_file = self.results_path / f"{scenario['attack_id']}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Analysis complete. Results saved to {results_file}")
        print(f"   Plots saved: {len(saved_plots)} files")
        
        return results
    
    def _states_to_sequences(self, states_df: pd.DataFrame) -> np.ndarray:
        """Convert state history to sequences for GenAI"""
        # Extract features
        features = ['expected_state', 'observed_state', 'divergence']
        available_features = [f for f in features if f in states_df.columns]
        
        if not available_features:
            # Fallback: use expected_state only
            data = states_df[['expected_state']].values
        else:
            data = states_df[available_features].values
        
        # Create sequences
        sequence_length = self.config['genai']['sequence_length']
        sequences = []
        
        for i in range(len(data) - sequence_length + 1):
            seq = data[i:i + sequence_length]
            sequences.append(seq)
        
        if len(sequences) == 0:
            # If not enough data, pad or repeat
            if len(data) > 0:
                seq = np.tile(data, (sequence_length, 1))[:sequence_length]
                sequences = [seq]
            else:
                sequences = [np.zeros((sequence_length, len(available_features) if available_features else 1))]
        
        return np.array(sequences)
    
    def _compute_tdi(self, states_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Trust Degradation Index from states"""
        tdi_scores = []
        
        for idx, row in states_df.iterrows():
            expected = row.get('expected_state', 0)
            observed = row.get('observed_state', 0)
            timestamp = row.get('timestamp', idx)
            
            trust = self.tdi.update_trust(timestamp, expected, observed)
            tdi_scores.append({
                'timestamp': timestamp,
                'trust_score': trust.trust_score,
                'trust_level': trust.trust_level.value
            })
        
        return pd.DataFrame(tdi_scores)
    
    def run_full_pipeline(self, data_file: Optional[str] = None,
                          train_model: bool = True,
                          run_attacks: bool = True) -> Dict:
        """
        Run complete pipeline
        
        Args:
            data_file: Path to SWaT data file (optional, uses dummy data if None)
            train_model: Whether to train GenAI model
            run_attacks: Whether to run attack scenarios
        
        Returns:
            Complete results
        """
        print("=" * 60)
        print("🛡️  GenAI-Powered Cyber-Aware Digital Twin")
        print("   IIT Kanpur Challenge Round – PS-6")
        print("=" * 60)
        
        results = {}
        
        # Process data
        if data_file and Path(data_file).exists():
            data = self.process_data(data_file)
            
            # Train model
            if train_model:
                train_seq, val_seq, test_seq = self.data_processor.split_data(
                    data['sequences']
                )
                history = self.train_genai_model(train_seq, val_seq)
                results['training_history'] = history
        else:
            print("⚠️  No data file provided. Using pre-trained model or generating synthetic data.")
            # Build model with dummy shape
            dummy_shape = (self.config['genai']['sequence_length'], len(self.config['data']['features']))
            self.genai.build_model(dummy_shape)
            # In production, would load pre-trained model
        
        # Run attack scenarios
        if run_attacks:
            print("\n" + "=" * 60)
            print("🎯 Running Attack Scenarios")
            print("=" * 60)
            
            attack_results = []
            
            # Run all attack types
            for attack_type in self.attack_generator.attack_types:
                try:
                    result = self.run_attack_scenario(attack_type)
                    attack_results.append(result)
                except Exception as e:
                    print(f"❌ Error running {attack_type}: {e}")
            
            results['attack_results'] = attack_results
            
            # Generate summary report
            self._generate_summary_report(attack_results)
        
        print("\n" + "=" * 60)
        print("✅ Pipeline Complete!")
        print("=" * 60)
        
        return results
    
    def _generate_summary_report(self, attack_results: List[Dict]):
        """Generate summary report"""
        print("\n📝 Generating Summary Report...")
        
        # Collect all gaps
        all_gaps = []
        for result in attack_results:
            if 'analysis' in result and 'gaps_identified' in result['analysis']:
                for gap in result['analysis']['gaps_identified']:
                    all_gaps.append(gap)
        
        # Create summary
        summary = {
            'total_attacks': len(attack_results),
            'total_gaps_identified': len(all_gaps),
            'unique_gap_categories': len(set(g.category for g in all_gaps)),
            'attacks_with_unsafe_states': sum(
                1 for r in attack_results
                if r.get('analysis', {}).get('unsafe_state_occurred', False)
            ),
            'average_detection_delay': np.mean([
                r.get('analysis', {}).get('detection_delay', float('inf'))
                for r in attack_results
                if r.get('analysis', {}).get('detection_delay') is not None
            ]) if any(r.get('analysis', {}).get('detection_delay') for r in attack_results) else None
        }
        
        # Save summary
        summary_file = self.report_path / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Summary report saved to {summary_file}")
        print(f"\n📊 Summary:")
        print(f"   Total Attacks: {summary['total_attacks']}")
        print(f"   Total Gaps Identified: {summary['total_gaps_identified']}")
        print(f"   Unique Gap Categories: {summary['unique_gap_categories']}")
        print(f"   Attacks with Unsafe States: {summary['attacks_with_unsafe_states']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GenAI-Powered Cyber-Aware Digital Twin Dashboard"
    )
    parser.add_argument(
        '--data-file', type=str,
        help='Path to SWaT dataset CSV file'
    )
    parser.add_argument(
        '--no-train', action='store_true',
        help='Skip model training'
    )
    parser.add_argument(
        '--no-attacks', action='store_true',
        help='Skip attack scenarios'
    )
    parser.add_argument(
        '--attack-type', type=str,
        choices=['sensor_spoofing', 'slow_manipulation', 'frozen_sensor', 'delayed_response'],
        help='Run specific attack type only'
    )
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = CyberAwareDashboard()
    
    # Run pipeline
    if args.attack_type:
        # Run single attack
        dashboard.run_attack_scenario(args.attack_type)
    else:
        # Run full pipeline
        dashboard.run_full_pipeline(
            data_file=args.data_file,
            train_model=not args.no_train,
            run_attacks=not args.no_attacks
        )


if __name__ == "__main__":
    main()
