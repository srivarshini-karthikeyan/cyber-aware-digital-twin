"""
Visualization Module

Creates plots and visualizations for:
- Time-series comparisons (Expected vs Observed)
- Attack window highlighting
- Unsafe state markers
- Trust Degradation Index
- Attack timeline
- Gap analysis tables
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml


class CyberAwareVisualizer:
    """
    Creates visualizations for cyber-aware digital twin analysis
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize visualizer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        vis_config = self.config['visualization']
        self.output_path = Path(vis_config['output_path'])
        self.figure_format = vis_config['figure_format']
        self.dpi = vis_config['dpi']
        self.style = vis_config['style']
        
        # Set style
        VALID_STYLES = ["white", "dark", "whitegrid", "darkgrid", "ticks"]

        if self.style not in VALID_STYLES:
            print(f"[WARN] Invalid seaborn style '{self.style}', defaulting to 'whitegrid'")
            self.style = "whitegrid"

        sns.set_style(self.style)

        plt.rcParams['figure.dpi'] = self.dpi
        
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def plot_three_state_comparison(self, states_df: pd.DataFrame,
                                    attack_window: Optional[Tuple[float, float]] = None,
                                    title: str = "Three-State Model Comparison",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Expected vs Observed vs Believed states
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot three states
        ax.plot(states_df['timestamp'], states_df['expected_state'],
                label='Expected State (Digital Twin)', linewidth=2, color='blue', linestyle='--')
        ax.plot(states_df['timestamp'], states_df['observed_state'],
                label='Observed State (Sensor)', linewidth=2, color='green')
        ax.plot(states_df['timestamp'], states_df['believed_state'],
                label='Believed State (Controller)', linewidth=2, color='orange', linestyle=':')
        
        # Highlight attack window
        if attack_window:
            ax.axvspan(attack_window[0], attack_window[1],
                      alpha=0.2, color='red', label='Attack Window')
        
        # Mark unsafe states
        unsafe_mask = states_df['safety_state'].isin(['unsafe', 'critical'])
        if unsafe_mask.any():
            unsafe_times = states_df.loc[unsafe_mask, 'timestamp']
            unsafe_levels = states_df.loc[unsafe_mask, 'observed_state']
            ax.scatter(unsafe_times, unsafe_levels, color='red', s=100,
                      marker='X', label='Unsafe State', zorder=5)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Tank Level (mm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_state_divergence(self, states_df: pd.DataFrame,
                             attack_window: Optional[Tuple[float, float]] = None,
                             title: str = "State Divergence Over Time",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot divergence between states
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot divergence
        ax.plot(states_df['timestamp'], states_df['divergence'],
                linewidth=2, color='red', label='Divergence')
        
        # Threshold line
        threshold = self.config['digital_twin']['state_comparison']['divergence_threshold']
        ax.axhline(y=threshold, color='orange', linestyle='--',
                  linewidth=2, label=f'Divergence Threshold ({threshold})')
        
        # Highlight attack window
        if attack_window:
            ax.axvspan(attack_window[0], attack_window[1],
                      alpha=0.2, color='red', label='Attack Window')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Divergence', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_trust_degradation_index(self, tdi_data: pd.DataFrame,
                                    title: str = "Trust Degradation Index (TDI)",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Trust Degradation Index over time
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot TDI
        ax.plot(tdi_data['timestamp'], tdi_data['trust_score'],
                linewidth=2, color='blue', label='Trust Score')
        
        # Trust level zones
        trust_levels = self.config['advanced_features']['trust_degradation_index']['trust_levels']
        ax.axhspan(trust_levels['green'], 1.0, alpha=0.2, color='green', label='Green Zone')
        ax.axhspan(trust_levels['yellow'], trust_levels['green'], alpha=0.2, color='yellow', label='Yellow Zone')
        ax.axhspan(0.0, trust_levels['yellow'], alpha=0.2, color='red', label='Red Zone')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Trust Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_attack_timeline(self, analysis_data: Dict,
                            title: str = "Attack Timeline Analysis",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot attack timeline with detection and unsafe state markers
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        attack_start = analysis_data['start_time']
        attack_end = analysis_data['end_time']
        detection_time = analysis_data.get('detection_time')
        unsafe_start = analysis_data.get('unsafe_state_start')
        unsafe_end = None
        if unsafe_start and analysis_data.get('unsafe_state_duration'):
            unsafe_end = unsafe_start + analysis_data['unsafe_state_duration']
        
        # Attack window
        ax.axvspan(attack_start, attack_end, alpha=0.3, color='red', label='Attack Active')
        
        # Detection marker
        if detection_time:
            ax.axvline(x=detection_time, color='orange', linestyle='--',
                      linewidth=2, label=f'Detection ({detection_time:.1f}s)')
            # Detection delay
            delay = detection_time - attack_start
            ax.annotate(f'Delay: {delay:.1f}s', xy=(detection_time, 0.5),
                       xytext=(detection_time + 5, 0.6),
                       arrowprops=dict(arrowstyle='->', color='orange'),
                       fontsize=10, fontweight='bold')
        
        # Unsafe state window
        if unsafe_start:
            if unsafe_end:
                ax.axvspan(unsafe_start, unsafe_end, alpha=0.5, color='darkred',
                          label='Unsafe State')
            else:
                ax.axvline(x=unsafe_start, color='darkred', linestyle=':',
                          linewidth=2, label='Unsafe State Start')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Event', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_unsafe_state_heatmap(self, states_df: pd.DataFrame,
                                  title: str = "Unsafe State Heatmap",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap showing unsafe states over time
        """
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Map safety states to numeric values
        safety_map = {'safe': 0, 'warning': 1, 'unsafe': 2, 'critical': 3}
        states_df['safety_numeric'] = states_df['safety_state'].map(safety_map)
        
        # Create heatmap data
        heatmap_data = states_df[['timestamp', 'safety_numeric']].set_index('timestamp').T
        
        # Plot heatmap
        sns.heatmap(heatmap_data, cmap='RdYlGn_r', cbar_kws={'label': 'Safety Level'},
                   yticklabels=['Safety State'], xticklabels=50, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_gap_analysis_table(self, gap_summary_df: pd.DataFrame,
                               title: str = "Cybersecurity Gap Analysis Summary",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create table visualization of gap analysis
        """
        fig, ax = plt.subplots(figsize=(16, max(6, len(gap_summary_df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=gap_summary_df.values,
                        colLabels=gap_summary_df.columns,
                        cellLoc='left',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(gap_summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style severity column
        if 'Severity' in gap_summary_df.columns:
            severity_col = gap_summary_df.columns.get_loc('Severity')
            for i in range(1, len(gap_summary_df) + 1):
                severity = gap_summary_df.iloc[i-1]['Severity']
                if severity == 'critical':
                    table[(i, severity_col)].set_facecolor('#f44336')
                elif severity == 'high':
                    table[(i, severity_col)].set_facecolor('#ff9800')
                elif severity == 'medium':
                    table[(i, severity_col)].set_facecolor('#ffeb3b')
                else:
                    table[(i, severity_col)].set_facecolor('#4caf50')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_comprehensive_dashboard(self, states_df: pd.DataFrame,
                                      analysis_data: Dict,
                                      gap_summary_df: pd.DataFrame,
                                      tdi_data: Optional[pd.DataFrame] = None,
                                      output_prefix: str = "dashboard") -> List[str]:
        """
        Create comprehensive dashboard with all visualizations
        
        Returns list of saved file paths
        """
        saved_files = []
        
        attack_window = (analysis_data['start_time'], analysis_data['end_time'])
        
        # 1. Three-state comparison
        fig1 = self.plot_three_state_comparison(
            states_df, attack_window,
            title="Three-State Model: Expected vs Observed vs Believed"
        )
        path1 = self.output_path / f"{output_prefix}_three_state.png"
        fig1.savefig(path1, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(str(path1))
        plt.close(fig1)
        
        # 2. State divergence
        fig2 = self.plot_state_divergence(
            states_df, attack_window,
            title="State Divergence: Cyber-Physical Gap"
        )
        path2 = self.output_path / f"{output_prefix}_divergence.png"
        fig2.savefig(path2, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(str(path2))
        plt.close(fig2)
        
        # 3. Attack timeline
        fig3 = self.plot_attack_timeline(
            analysis_data,
            title="Attack Timeline: Detection and Impact"
        )
        path3 = self.output_path / f"{output_prefix}_timeline.png"
        fig3.savefig(path3, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(str(path3))
        plt.close(fig3)
        
        # 4. Unsafe state heatmap
        fig4 = self.plot_unsafe_state_heatmap(
            states_df,
            title="Safety State Heatmap"
        )
        path4 = self.output_path / f"{output_prefix}_heatmap.png"
        fig4.savefig(path4, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(str(path4))
        plt.close(fig4)
        
        # 5. Gap analysis table
        if not gap_summary_df.empty:
            fig5 = self.plot_gap_analysis_table(
                gap_summary_df,
                title="Cybersecurity Gap Analysis"
            )
            path5 = self.output_path / f"{output_prefix}_gaps.png"
            fig5.savefig(path5, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(str(path5))
            plt.close(fig5)
        
        # 6. Trust Degradation Index
        if tdi_data is not None and not tdi_data.empty:
            fig6 = self.plot_trust_degradation_index(
                tdi_data,
                title="Trust Degradation Index (TDI)"
            )
            path6 = self.output_path / f"{output_prefix}_tdi.png"
            fig6.savefig(path6, format=self.figure_format, dpi=self.dpi, bbox_inches='tight')
            saved_files.append(str(path6))
            plt.close(fig6)
        
        return saved_files


if __name__ == "__main__":
    # Example usage
    visualizer = CyberAwareVisualizer()
    
    # Dummy data
    states_df = pd.DataFrame({
        'timestamp': range(100),
        'expected_state': 500 + np.random.randn(100) * 10,
        'observed_state': 500 + np.random.randn(100) * 10,
        'believed_state': 500 + np.random.randn(100) * 10,
        'divergence': np.random.rand(100) * 0.1,
        'safety_state': ['safe'] * 50 + ['unsafe'] * 30 + ['safe'] * 20
    })
    
    # Create plot
    fig = visualizer.plot_three_state_comparison(
        states_df, attack_window=(20, 50),
        title="Example Three-State Comparison"
    )
    plt.show()
