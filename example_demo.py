"""
Example Demo Script

Quick demonstration of the GenAI-Powered Cyber-Aware Digital Twin system
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.dashboard import CyberAwareDashboard
from src.attack_generator import AttackGenerator
from src.gap_analyzer import CyberGapAnalyzer
from src.visualizer import CyberAwareVisualizer


def demo_sensor_spoofing_attack():
    """Demonstrate sensor spoofing attack scenario"""
    print("=" * 60)
    print("🎯 DEMO: Sensor Spoofing Attack")
    print("=" * 60)
    
    # Initialize components
    dashboard = CyberAwareDashboard()
    
    # Run attack scenario
    results = dashboard.run_attack_scenario("sensor_spoofing")
    
    # Display results
    print("\n📊 Attack Analysis Results:")
    print(f"   Attack ID: {results['scenario']['attack_id']}")
    print(f"   Attack Type: {results['scenario']['attack_type']}")
    print(f"   Detected: {results['analysis']['detected']}")
    print(f"   Detection Delay: {results['analysis']['detection_delay']:.2f} seconds")
    print(f"   Unsafe State Occurred: {results['analysis']['unsafe_state_occurred']}")
    print(f"   Gaps Identified: {len(results['analysis']['gaps_identified'])}")
    print(f"   Risk Score: {results['analysis']['risk_score']:.2f}")
    
    # Display gaps
    print("\n🔍 Identified Gaps:")
    for i, gap in enumerate(results['analysis']['gaps_identified'], 1):
        print(f"   {i}. {gap.category} ({gap.severity})")
        print(f"      Description: {gap.description}")
        print(f"      Mitigation: {gap.mitigation}")

    
    # Display mitigations
    print("\n🛡️ Recommended Mitigations:")
    for i, mitigation in enumerate(results['analysis']['recommended_mitigations'], 1):
        print(f"   {i}. {mitigation}")
    
    print(f"\n📁 Results saved to: {results.get('plots', [])}")
    print("=" * 60)


def demo_all_attacks():
    """Demonstrate all attack scenarios"""
    print("=" * 60)
    print("🎯 DEMO: All Attack Scenarios")
    print("=" * 60)
    
    dashboard = CyberAwareDashboard()
    
    attack_types = [
        "sensor_spoofing",
        "slow_manipulation",
        "frozen_sensor",
        "delayed_response"
    ]
    
    all_results = []
    
    for attack_type in attack_types:
        print(f"\n🔴 Running {attack_type}...")
        try:
            results = dashboard.run_attack_scenario(attack_type)
            all_results.append(results)
            
            print(f"   ✅ Detected: {results['analysis']['detected']}")
            print(f"   ⏱️  Detection Delay: {results['analysis'].get('detection_delay', 'N/A')}")
            print(f"   ⚠️  Unsafe State: {results['analysis']['unsafe_state_occurred']}")
            print(f"   🔍 Gaps: {len(results['analysis']['gaps_identified'])}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary Statistics")
    print("=" * 60)
    
    total_gaps = sum(len(r['analysis']['gaps_identified']) for r in all_results)
    detected_attacks = sum(1 for r in all_results if r['analysis']['detected'])
    unsafe_attacks = sum(1 for r in all_results if r['analysis']['unsafe_state_occurred'])
    
    print(f"   Total Attacks: {len(all_results)}")
    print(f"   Detected Attacks: {detected_attacks}")
    print(f"   Attacks with Unsafe States: {unsafe_attacks}")
    print(f"   Total Gaps Identified: {total_gaps}")
    
    avg_delay = np.mean([
        r['analysis'].get('detection_delay', 0)
        for r in all_results
        if r['analysis'].get('detection_delay') is not None
    ])
    print(f"   Average Detection Delay: {avg_delay:.2f} seconds")
    
    print("=" * 60)


def demo_trust_degradation():
    """Demonstrate Trust Degradation Index"""
    print("=" * 60)
    print("🎯 DEMO: Trust Degradation Index (TDI)")
    print("=" * 60)
    
    from src.advanced_features import TrustDegradationIndex
    
    tdi = TrustDegradationIndex()
    
    # Simulate trust degradation during attack
    print("\n📉 Simulating Trust Degradation During Attack...")
    
    normal_level = 500.0
    
    for t in range(100):
        if t < 20:
            # Normal operation
            observed = normal_level + np.random.randn() * 5
        elif t < 80:
            # Attack phase - sensor spoofing
            actual = normal_level + (t - 20) * 2.0
            observed = actual * 0.3  # Spoofed to 30%
        else:
            # Recovery
            observed = normal_level + np.random.randn() * 5
        
        trust = tdi.update_trust(float(t), normal_level, observed)
        
        if t % 20 == 0:
            print(f"   t={t:3d}s: Trust={trust.trust_score:.3f} ({trust.trust_level.value})")
    
    # Get trust history
    tdi_df = tdi.get_trust_history_df()
    print(f"\n📊 Trust History Summary:")
    print(f"   Total Samples: {len(tdi_df)}")
    print(f"   Average Trust: {tdi_df['trust_score'].mean():.3f}")
    print(f"   Min Trust: {tdi_df['trust_score'].min():.3f}")
    print(f"   Red Zone Samples: {(tdi_df['trust_level'] == 'red').sum()}")
    
    print("=" * 60)


def demo_gap_analysis_table():
    """Demonstrate gap analysis table generation"""
    print("=" * 60)
    print("🎯 DEMO: Gap Analysis Table")
    print("=" * 60)
    
    analyzer = CyberGapAnalyzer()
    
    # Create dummy attack data
    attack_data = {
        'attack_id': 'demo_001',
        'attack_type': 'sensor_spoofing',
        'start_time': 10.0,
        'end_time': 50.0,
        'duration': 40.0
    }
    
    # Create dummy states
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
    
    # Get gap summary table
    gap_table = analyzer.get_gap_summary_table()
    
    print("\n📋 Gap Analysis Summary Table:")
    print(gap_table.to_string(index=False))
    
    print("\n🔍 Detailed Gaps:")
    for gap in analysis.gaps_identified:
        print(f"   • {gap.category.value} ({gap.severity.value})")
        print(f"     {gap.description}")
        print(f"     Mitigation: {gap.mitigation}")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🛡️  GenAI-Powered Cyber-Aware Digital Twin")
    print("   Example Demo Script")
    print("=" * 60)
    
    # Run demos
    print("\n1️⃣ Sensor Spoofing Attack Demo")
    demo_sensor_spoofing_attack()
    
    print("\n2️⃣ Trust Degradation Index Demo")
    demo_trust_degradation()
    
    print("\n3️⃣ Gap Analysis Table Demo")
    demo_gap_analysis_table()
    
    print("\n4️⃣ All Attacks Demo")
    demo_all_attacks()
    
    print("\n✅ All demos complete!")
    print("=" * 60)
