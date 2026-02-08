"""
Publication-Grade Output Generator

Creates publication-quality tables, plots, and reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import yaml


class PublicationOutputGenerator:
    """
    Generates publication-grade outputs for research presentation
    """
    
    def __init__(self, output_dir: str = "outputs/publication"):
        """Initialize output generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
    def generate_metrics_table(self, metrics_df: pd.DataFrame,
                               title: str = "Detection Performance Metrics",
                               filename: Optional[str] = None) -> str:
        """
        Generate publication-quality metrics table
        
        Args:
            metrics_df: DataFrame with metrics
            title: Table title
            filename: Output filename (optional)
        
        Returns:
            Path to saved table
        """
        if filename is None:
            filename = f"metrics_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(metrics_df) * 0.4)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style cells
        for i in range(1, len(metrics_df) + 1):
            for j in range(len(metrics_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def generate_performance_comparison_plot(self, per_attack_metrics: Dict,
                                            filename: Optional[str] = None) -> str:
        """
        Generate performance comparison plot across attack types
        
        Args:
            per_attack_metrics: Dictionary mapping attack type to metrics
            filename: Output filename (optional)
        
        Returns:
            Path to saved plot
        """
        if filename is None:
            filename = f"performance_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        attack_types = list(per_attack_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        data = []
        for attack_type in attack_types:
            m = per_attack_metrics[attack_type]
            data.append({
                'Attack Type': attack_type,
                'Accuracy': m.accuracy,
                'Precision': m.precision,
                'Recall': m.recall,
                'F1-Score': m.f1_score,
                'ROC-AUC': m.roc_auc
            })
        
        df = pd.DataFrame(data)
        df_melted = df.melt(id_vars='Attack Type', var_name='Metric', value_name='Score')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Attack Type', y='Score', hue='Metric', ax=ax)
        ax.set_title('Detection Performance Across Attack Types', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.legend(title='Metric', loc='best')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def generate_latency_analysis_plot(self, latency_data: Dict,
                                      filename: Optional[str] = None) -> str:
        """
        Generate detection latency analysis plot
        
        Args:
            latency_data: Dictionary with latency data per attack type
            filename: Output filename (optional)
        
        Returns:
            Path to saved plot
        """
        if filename is None:
            filename = f"latency_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        attack_types = list(latency_data.keys())
        means = [latency_data[at]['mean'] for at in attack_types]
        stds = [latency_data[at]['std'] for at in attack_types]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(attack_types))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Detection Latency (seconds)', fontsize=12)
        ax.set_title('Detection Latency by Attack Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(attack_types, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.5, f'{mean:.2f}s', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def generate_experiment_summary(self, experiment_data: Dict,
                                   filename: Optional[str] = None) -> str:
        """
        Generate experiment summary JSON
        
        Args:
            experiment_data: Dictionary with experiment results
            filename: Output filename (optional)
        
        Returns:
            Path to saved JSON
        """
        if filename is None:
            filename = f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def generate_csv_results(self, data: pd.DataFrame,
                            filename: Optional[str] = None) -> str:
        """
        Generate CSV results file
        
        Args:
            data: DataFrame to save
            filename: Output filename (optional)
        
        Returns:
            Path to saved CSV
        """
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.output_dir / filename
        data.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def generate_comprehensive_report(self, metrics: Dict,
                                     per_attack_metrics: Dict,
                                     explanations: List,
                                     mitigations: List,
                                     filename: Optional[str] = None) -> str:
        """
        Generate comprehensive research report
        
        Args:
            metrics: Overall metrics
            per_attack_metrics: Per-attack metrics
            explanations: List of explanations
            mitigations: List of mitigations
            filename: Output filename (optional)
        
        Returns:
            Path to saved report
        """
        if filename is None:
            filename = f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("# Comprehensive Detection Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall metrics
            f.write("## Overall Performance Metrics\n\n")
            f.write(f"- Accuracy: {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"- Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"- Recall: {metrics.get('recall', 0):.4f}\n")
            f.write(f"- F1-Score: {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"- ROC-AUC: {metrics.get('roc_auc', 0):.4f}\n")
            f.write(f"- Average Detection Latency: {metrics.get('detection_latency_mean', 0):.2f}s\n\n")
            
            # Per-attack metrics
            f.write("## Per-Attack Performance\n\n")
            for attack_type, attack_metrics in per_attack_metrics.items():
                f.write(f"### {attack_type}\n\n")
                f.write(f"- Accuracy: {attack_metrics.accuracy:.4f}\n")
                f.write(f"- Precision: {attack_metrics.precision:.4f}\n")
                f.write(f"- Recall: {attack_metrics.recall:.4f}\n")
                f.write(f"- F1-Score: {attack_metrics.f1_score:.4f}\n")
                f.write(f"- Detection Latency: {attack_metrics.detection_latency_mean:.2f}s\n\n")
            
            # Explanations
            f.write("## Anomaly Explanations\n\n")
            for i, exp in enumerate(explanations[:10], 1):  # Top 10
                f.write(f"### Anomaly {i}\n\n")
                if hasattr(exp, 'human_readable'):
                    f.write(f"{exp.human_readable}\n\n")
            
            # Mitigations
            f.write("## Recommended Mitigations\n\n")
            for i, mit in enumerate(mitigations[:10], 1):  # Top 10
                f.write(f"{i}. **{mit.action.value}** ({mit.priority.value})\n")
                f.write(f"   {mit.description}\n\n")
        
        return str(filepath)


if __name__ == "__main__":
    # Example usage
    generator = PublicationOutputGenerator()
    
    # Dummy metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Overall': [0.95, 0.92, 0.88]
    })
    
    # Generate table
    path = generator.generate_metrics_table(metrics_df)
    print(f"Generated table: {path}")
