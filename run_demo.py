"""
Quick Demo Runner - See Everything in Action!

This script demonstrates all the research-grade features
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.research_dashboard import ResearchGradeDashboard
from src.attack_generator import AttackGenerator
from src.visualizer import CyberAwareVisualizer

def main():
    print("=" * 70)
    print("🛡️  RESEARCH-GRADE CYBER-AWARE DIGITAL TWIN - DEMO")
    print("=" * 70)
    print()
    
    # Initialize dashboard
    print("📦 Initializing Research-Grade Dashboard...")
    dashboard = ResearchGradeDashboard()
    print("✅ Dashboard initialized!")
    print()
    
    # Generate some dummy training data
    print("🧠 Training Ensemble Models (with dummy data)...")
    dummy_sequences = np.random.rand(100, 60, 3)  # 100 samples, 60 timesteps, 3 features
    dummy_features = dummy_sequences.reshape(100, -1)
    
    try:
        dashboard.train_ensemble_models(dummy_sequences, dummy_features)
        print("✅ Models trained!")
    except Exception as e:
        print(f"⚠️ Training error (expected with dummy data): {e}")
        print("   Continuing with demo...")
    print()
    
    # Simulate real-time stream
    print("🔄 Simulating Real-Time Stream Processing...")
    print("   Processing 50 samples...")
    print()
    
    results = []
    for i in range(50):
        # Simulate sensor data
        if i < 10:
            # Normal operation
            sensor_data = {
                'level': 500.0 + np.random.randn() * 5,
                'valve': 1,
                'pump': 0
            }
            ground_truth = False
            attack_type = None
        elif i < 40:
            # Attack phase - sensor spoofing
            actual_level = 500.0 + i * 2.0
            sensor_data = {
                'level': actual_level * 0.3,  # Spoofed to 30%
                'valve': 1,
                'pump': 0
            }
            ground_truth = True
            attack_type = "sensor_spoofing"
        else:
            # Recovery
            sensor_data = {
                'level': 500.0 + np.random.randn() * 5,
                'valve': 1,
                'pump': 0
            }
            ground_truth = False
            attack_type = None
        
        # Process stream
        result = dashboard.process_real_time_stream(
            sensor_data, 
            ground_truth=ground_truth,
            attack_type=attack_type
        )
        
        results.append(result)
        
        # Print every 10th result
        if i % 10 == 0:
            if result.get('ready'):
                print(f"   Sample {i}: Anomaly={result.get('anomaly_detected', False)}, "
                      f"Confidence={result.get('confidence', 0):.2f}, "
                      f"Trust={result.get('trust_score', 0):.2f}")
    
    print()
    print("✅ Stream processing complete!")
    print()
    
    # Show statistics
    print("📊 Statistics:")
    stream_stats = dashboard.streaming_processor.get_statistics()
    print(f"   Total samples processed: {stream_stats['total_samples']}")
    print(f"   Anomalies detected: {stream_stats['anomalies_detected']}")
    print(f"   Average detection latency: {stream_stats['avg_detection_latency']:.2f}s")
    print()
    
    # Show trust summary
    print("🔒 Trust Assessment Summary:")
    trust_summary = dashboard.trust_assessment.get_trust_summary()
    if trust_summary:
        print(f"   Current trust: {trust_summary.get('current_trust', 0):.2f}")
        print(f"   Trust state: {trust_summary.get('current_state', 'unknown')}")
        print(f"   Attack periods: {trust_summary.get('attack_periods', 0)}")
        print(f"   Recovery periods: {trust_summary.get('recovery_periods', 0)}")
    print()
    
    # Generate attack scenario
    print("🎯 Generating Attack Scenario...")
    attack_gen = AttackGenerator()
    scenario = attack_gen.generate_sensor_spoofing_attack(duration=30)
    print(f"✅ Generated: {scenario['attack_id']}")
    print(f"   Attack type: {scenario['attack_type']}")
    print(f"   Duration: {scenario['duration']}s")
    print()
    
    # Show validation metrics (if available)
    print("📈 Validation Metrics:")
    try:
        overall_metrics = dashboard.validation_metrics.compute_metrics()
        print(f"   Accuracy: {overall_metrics.accuracy:.4f}")
        print(f"   Precision: {overall_metrics.precision:.4f}")
        print(f"   Recall: {overall_metrics.recall:.4f}")
        print(f"   F1-Score: {overall_metrics.f1_score:.4f}")
    except:
        print("   (Run validation experiment for full metrics)")
    print()
    
    # Show adaptive threshold
    print("🎚️ Adaptive Threshold:")
    threshold_state = dashboard.adaptive_threshold.get_threshold_state()
    print(f"   Current threshold: {threshold_state['current_threshold']:.4f}")
    print(f"   Drift detected: {dashboard.adaptive_threshold.drift_detected}")
    print()
    
    # Show ensemble info
    print("🤖 Ensemble Detector:")
    print(f"   Models: LSTM, Isolation Forest, Statistical, Density-based")
    print(f"   Fusion weights: {dashboard.ensemble_detector.weights}")
    print()
    
    print("=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("📝 Next Steps:")
    print("   1. Run full validation: python run_validation.py")
    print("   2. Generate attack scenarios: python example_demo.py")
    print("   3. View visualizations: Check outputs/plots/")
    print("   4. Read documentation: RESEARCH_GRADE_UPGRADE.md")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
