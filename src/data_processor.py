"""
Data Preprocessing Layer for SWaT Dataset
Handles feature selection, normalization, and time-series alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class SWaTDataProcessor:
    """
    Processes SWaT dataset for Raw Water Tank Level Control System.
    Focuses on: LIT101 (level), MV101 (valve), P101 (pump)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.features = self.config['data']['features']
        self.time_window = self.config['data']['time_window']
        
    def load_swat_data(self, file_path: str) -> pd.DataFrame:
        """
        Load SWaT dataset CSV file
        
        Expected columns:
        - Timestamp
        - LIT101 (Tank Level in mm)
        - MV101 (Inlet Valve: 0=closed, 1=open)
        - P101 (Outlet Pump: 0=OFF, 1=ON)
        - Attack (0=normal, 1=attack)
        """
        df = pd.read_csv(file_path)
        
        # Convert timestamp if needed
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        elif 'Time' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Time'])
        
        return df
    
    def select_subsystem_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only features relevant to Raw Water Tank subsystem"""
        required_cols = ['Timestamp'] + self.features
        
        # Check if Attack column exists (for evaluation only)
        if 'Attack' in df.columns:
            required_cols.append('Attack')
        
        # Select available columns
        available_cols = [col for col in required_cols if col in df.columns]
        df_subset = df[available_cols].copy()
        
        return df_subset
    
    def normalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features to [0, 1] range
        Returns normalized dataframe and normalization parameters
        """
        df_norm = df.copy()
        norm_params = {}
        
        for feature in self.features:
            if feature in df_norm.columns:
                min_val = df_norm[feature].min()
                max_val = df_norm[feature].max()
                
                if max_val > min_val:
                    df_norm[feature] = (df_norm[feature] - min_val) / (max_val - min_val)
                    norm_params[feature] = {'min': min_val, 'max': max_val}
                else:
                    norm_params[feature] = {'min': min_val, 'max': max_val + 1}
        
        return df_norm, norm_params
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = None) -> np.ndarray:
        """
        Create time-series sequences for LSTM/VAE models
        
        Returns: (n_samples, sequence_length, n_features)
        """
        if sequence_length is None:
            sequence_length = self.time_window
        
        sequences = []
        feature_data = df[self.features].values
        
        for i in range(len(feature_data) - sequence_length + 1):
            seq = feature_data[i:i + sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def separate_normal_attack(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate normal and attack data
        Note: Labels used only for evaluation, not training
        """
        if 'Attack' not in df.columns:
            # If no attack label, assume all normal
            return df.copy(), pd.DataFrame()
        
        normal_df = df[df['Attack'] == 0].copy()
        attack_df = df[df['Attack'] == 1].copy()
        
        return normal_df, attack_df
    
    def prepare_training_data(self, file_path: str) -> Dict:
        """
        Complete preprocessing pipeline for training
        
        Returns:
        {
            'sequences': np.array,
            'normal_data': pd.DataFrame,
            'attack_data': pd.DataFrame,
            'norm_params': dict,
            'metadata': dict
        }
        """
        # Load data
        df = self.load_swat_data(file_path)
        
        # Select subsystem features
        df_subset = self.select_subsystem_features(df)
        
        # Separate normal and attack
        normal_df, attack_df = self.separate_normal_attack(df_subset)
        
        # Normalize (using normal data statistics only)
        normal_df_norm, norm_params = self.normalize_features(normal_df)
        
        # Create sequences from normal data
        sequences = self.create_sequences(normal_df_norm)
        
        # Normalize attack data using same parameters
        if not attack_df.empty:
            attack_df_norm = attack_df.copy()
            for feature in self.features:
                if feature in norm_params:
                    params = norm_params[feature]
                    attack_df_norm[feature] = (
                        (attack_df[feature] - params['min']) / 
                        (params['max'] - params['min'])
                    )
        else:
            attack_df_norm = attack_df
        
        metadata = {
            'total_samples': len(df),
            'normal_samples': len(normal_df),
            'attack_samples': len(attack_df),
            'sequence_length': self.time_window,
            'n_features': len(self.features)
        }
        
        return {
            'sequences': sequences,
            'normal_data': normal_df_norm,
            'attack_data': attack_df_norm,
            'norm_params': norm_params,
            'metadata': metadata
        }
    
    def split_data(self, sequences: np.ndarray, 
                   train_split: float = 0.7,
                   val_split: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train/val/test sets
        """
        n_samples = len(sequences)
        train_end = int(n_samples * train_split)
        val_end = train_end + int(n_samples * val_split)
        
        train_seq = sequences[:train_end]
        val_seq = sequences[train_end:val_end]
        test_seq = sequences[val_end:]
        
        return train_seq, val_seq, test_seq


if __name__ == "__main__":
    # Example usage
    processor = SWaTDataProcessor()
    
    # Assuming SWaT data file exists
    # data = processor.prepare_training_data("data/raw/swat_data.csv")
    # print(f"Prepared {len(data['sequences'])} sequences")
    # print(f"Normal samples: {data['metadata']['normal_samples']}")
    # print(f"Attack samples: {data['metadata']['attack_samples']}")
