"""
Enhanced Trust & Reliability Assessment

Tracks trust degradation during attacks and recovery post-attack
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque
import yaml


class TrustState(Enum):
    """Trust state classification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class TrustSnapshot:
    """Trust state snapshot"""
    timestamp: float
    trust_score: float  # [0, 1]
    trust_state: TrustState
    degradation_rate: float
    recovery_rate: float
    component: str
    attack_active: bool


class EnhancedTrustAssessment:
    """
    Enhanced trust assessment with recovery tracking
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize enhanced trust assessment"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        trust_config = self.config['advanced_features']['trust_degradation_index']
        self.trust_levels = trust_config['trust_levels']
        
        self.trust_history: List[TrustSnapshot] = []
        self.attack_periods: List[Tuple[float, float]] = []  # (start, end)
        self.recovery_periods: List[Tuple[float, float]] = []  # (start, end)
        
        # Trust statistics
        self.stats = {
            'min_trust_during_attack': 1.0,
            'max_degradation_rate': 0.0,
            'recovery_time': 0.0,
            'trust_recovery_rate': 0.0
        }
    
    def update_trust(self, timestamp: float, trust_score: float,
                    component: str = "Level Sensor",
                    attack_active: bool = False) -> TrustSnapshot:
        """
        Update trust assessment
        
        Args:
            timestamp: Current timestamp
            trust_score: Current trust score [0, 1]
            component: Component name
            attack_active: Whether attack is currently active
        
        Returns:
            TrustSnapshot object
        """
        # Classify trust state
        if trust_score >= self.trust_levels['green']:
            trust_state = TrustState.HIGH
        elif trust_score >= self.trust_levels['yellow']:
            trust_state = TrustState.MEDIUM
        elif trust_score >= self.trust_levels['red']:
            trust_state = TrustState.LOW
        else:
            trust_state = TrustState.CRITICAL
        
        # Compute degradation/recovery rate
        degradation_rate = 0.0
        recovery_rate = 0.0
        
        if len(self.trust_history) > 0:
            prev_trust = self.trust_history[-1].trust_score
            time_diff = timestamp - self.trust_history[-1].timestamp
            
            if time_diff > 0:
                trust_change = trust_score - prev_trust
                rate = trust_change / time_diff
                
                if rate < 0:
                    degradation_rate = abs(rate)
                else:
                    recovery_rate = rate
        
        # Track attack periods
        if attack_active and len(self.attack_periods) == 0:
            # Start new attack period
            self.attack_periods.append((timestamp, timestamp))
        elif attack_active and len(self.attack_periods) > 0:
            # Update current attack period
            self.attack_periods[-1] = (self.attack_periods[-1][0], timestamp)
        elif not attack_active and len(self.attack_periods) > 0 and \
             self.attack_periods[-1][1] == timestamp - 1:
            # Attack just ended, start recovery period
            self.recovery_periods.append((timestamp, timestamp))
        elif not attack_active and len(self.recovery_periods) > 0:
            # Update recovery period
            self.recovery_periods[-1] = (self.recovery_periods[-1][0], timestamp)
        
        # Update statistics
        if attack_active:
            self.stats['min_trust_during_attack'] = min(
                self.stats['min_trust_during_attack'], trust_score
            )
            self.stats['max_degradation_rate'] = max(
                self.stats['max_degradation_rate'], degradation_rate
            )
        
        # Compute recovery statistics
        if len(self.recovery_periods) > 0 and not attack_active:
            recovery_start = self.recovery_periods[-1][0]
            if len(self.trust_history) > 0:
                recovery_start_trust = self.trust_history[-1].trust_score
                current_trust = trust_score
                recovery_time = timestamp - recovery_start
                
                if recovery_time > 0:
                    self.stats['recovery_time'] = recovery_time
                    self.stats['trust_recovery_rate'] = (
                        current_trust - recovery_start_trust
                    ) / recovery_time
        
        snapshot = TrustSnapshot(
            timestamp=timestamp,
            trust_score=trust_score,
            trust_state=trust_state,
            degradation_rate=degradation_rate,
            recovery_rate=recovery_rate,
            component=component,
            attack_active=attack_active
        )
        
        self.trust_history.append(snapshot)
        return snapshot
    
    def get_trust_summary(self) -> Dict:
        """
        Get comprehensive trust summary
        
        Returns:
            Dictionary with trust statistics
        """
        if len(self.trust_history) == 0:
            return {}
        
        trust_scores = [t.trust_score for t in self.trust_history]
        
        summary = {
            'current_trust': self.trust_history[-1].trust_score,
            'current_state': self.trust_history[-1].trust_state.value,
            'average_trust': np.mean(trust_scores),
            'min_trust': np.min(trust_scores),
            'max_trust': np.max(trust_scores),
            'trust_std': np.std(trust_scores),
            'total_snapshots': len(self.trust_history),
            'attack_periods': len(self.attack_periods),
            'recovery_periods': len(self.recovery_periods),
            **self.stats
        }
        
        return summary
    
    def get_trust_evolution_df(self) -> pd.DataFrame:
        """Convert trust history to DataFrame"""
        if not self.trust_history:
            return pd.DataFrame()
        
        data = {
            'timestamp': [t.timestamp for t in self.trust_history],
            'trust_score': [t.trust_score for t in self.trust_history],
            'trust_state': [t.trust_state.value for t in self.trust_history],
            'degradation_rate': [t.degradation_rate for t in self.trust_history],
            'recovery_rate': [t.recovery_rate for t in self.trust_history],
            'component': [t.component for t in self.trust_history],
            'attack_active': [t.attack_active for t in self.trust_history]
        }
        
        return pd.DataFrame(data)
    
    def plot_trust_evolution(self, save_path: Optional[str] = None):
        """
        Plot trust evolution over time
        
        Args:
            save_path: Path to save plot (optional)
        
        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt
        
        df = self.get_trust_evolution_df()
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot trust score
        ax.plot(df['timestamp'], df['trust_score'], 
               linewidth=2, color='blue', label='Trust Score')
        
        # Highlight attack periods
        for start, end in self.attack_periods:
            ax.axvspan(start, end, alpha=0.2, color='red', label='Attack Period')
        
        # Highlight recovery periods
        for start, end in self.recovery_periods:
            ax.axvspan(start, end, alpha=0.2, color='green', label='Recovery Period')
        
        # Trust level zones
        ax.axhspan(self.trust_levels['green'], 1.0, alpha=0.1, color='green')
        ax.axhspan(self.trust_levels['yellow'], self.trust_levels['green'], 
                  alpha=0.1, color='yellow')
        ax.axhspan(0.0, self.trust_levels['yellow'], alpha=0.1, color='red')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Trust Score', fontsize=12)
        ax.set_title('Trust Evolution with Attack and Recovery Periods', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def reset(self):
        """Reset trust assessment"""
        self.trust_history = []
        self.attack_periods = []
        self.recovery_periods = []
        self.stats = {
            'min_trust_during_attack': 1.0,
            'max_degradation_rate': 0.0,
            'recovery_time': 0.0,
            'trust_recovery_rate': 0.0
        }


if __name__ == "__main__":
    # Example usage
    trust = EnhancedTrustAssessment()
    
    # Simulate trust evolution
    for t in range(200):
        if t < 50:
            trust_score = 0.9 - np.random.randn() * 0.05  # Normal
            attack = False
        elif t < 150:
            trust_score = 0.9 - (t - 50) * 0.01  # Degrading
            attack = True
        else:
            trust_score = 0.2 + (t - 150) * 0.01  # Recovering
            attack = False
        
        trust_score = np.clip(trust_score, 0.0, 1.0)
        trust.update_trust(float(t), trust_score, attack_active=attack)
    
    # Get summary
    summary = trust.get_summary()
    print("Trust Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
