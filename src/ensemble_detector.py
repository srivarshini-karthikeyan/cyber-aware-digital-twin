"""
Multi-Model Ensemble Intelligence

Combines multiple analytical perspectives:
1. Deep learning (LSTM Autoencoder)
2. Statistical deviation analysis
3. Isolation-based anomaly detection
4. Density-based clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import yaml

try:
    from .genai_engine import LSTMAutoencoder
except ImportError:
    # Fallback for direct import
    from genai_engine import LSTMAutoencoder


class EnsembleAnomalyDetector:
    """
    Ensemble of multiple anomaly detection models
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ensemble detector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model 1: Deep Learning (LSTM Autoencoder)
        self.lstm_model = LSTMAutoencoder(config_path)
        
        # Model 2: Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        # Model 3: Statistical Z-score detector
        self.statistical_scaler = StandardScaler()
        self.statistical_mean = None
        self.statistical_std = None
        self.z_threshold = 3.0  # 3-sigma rule
        
        # Model 4: Density-based (Local Outlier Factor)
        from sklearn.neighbors import LocalOutlierFactor
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        
        # Ensemble fusion weights (adaptive)
        self.weights = {
            'lstm': 0.4,
            'isolation': 0.25,
            'statistical': 0.2,
            'density': 0.15
        }
        
        self.is_trained = False
        self.training_data = None
    
    def train_ensemble(self, normal_sequences: np.ndarray,
                      normal_features: Optional[np.ndarray] = None):
        """
        Train all ensemble models on normal data
        
        Args:
            normal_sequences: Time-series sequences for LSTM
            normal_features: Feature vectors for other models
        """
        print("🧠 Training Ensemble Models...")
        
        # Train LSTM Autoencoder
        if normal_sequences is not None and len(normal_sequences) > 0:
            print("  Training LSTM Autoencoder...")
            self.lstm_model.build_model((normal_sequences.shape[1], normal_sequences.shape[2]))
            # Use subset for quick training in demo
            train_size = min(100, len(normal_sequences))
            self.lstm_model.train(normal_sequences[:train_size], verbose=0)
            self.lstm_model.is_trained = True
            self.lstm_model.trained = True
        
        # Prepare features for other models
        if normal_features is None and normal_sequences is not None:
            # Flatten sequences to features
            normal_features = normal_sequences.reshape(
                normal_sequences.shape[0], -1
            )
        
        if normal_features is not None and len(normal_features) > 0:
            # Train Isolation Forest
            print("  Training Isolation Forest...")
            self.isolation_forest.fit(normal_features)
            
            # Train Statistical model
            print("  Training Statistical Z-score detector...")
            self.statistical_scaler.fit(normal_features)
            self.statistical_mean = np.mean(normal_features, axis=0)
            self.statistical_std = np.std(normal_features, axis=0)
            
            # Train LOF
            print("  Training Local Outlier Factor...")
            train_size = min(100, len(normal_features))
            self.lof.fit(normal_features[:train_size])
        
        self.is_trained = True
        self.training_data = normal_sequences if normal_sequences is not None else normal_features
        print("✅ Ensemble training complete")
    
    def detect_anomaly_ensemble(self, sequences: np.ndarray,
                               features: Optional[np.ndarray] = None) -> Dict:
        """
        Detect anomalies using ensemble of models
        
        Args:
            sequences: Time-series sequences (for LSTM)
            features: Feature vectors (for other models)
        
        Returns:
            Dictionary with ensemble results
        """
        if not self.is_trained:
            # Auto-train if not trained
            if sequences is not None:
                self.train_ensemble(sequences, features)
            else:
                raise ValueError("Models not trained and no training data provided")
        
        results = {}
        
        # Model 1: LSTM Autoencoder
        if sequences is not None and len(sequences) > 0:
            try:
                lstm_flags, lstm_scores = self.lstm_model.detect_anomalies(sequences)
                results['lstm'] = {
                    'anomaly_flags': lstm_flags,
                    'confidence_scores': lstm_scores,
                    'anomaly_rate': np.mean(lstm_flags)
                }
            except Exception as e:
                print(f"⚠️ LSTM detection error: {e}")
                results['lstm'] = {
                    'anomaly_flags': np.zeros(len(sequences), dtype=bool),
                    'confidence_scores': np.zeros(len(sequences)),
                    'anomaly_rate': 0.0
                }
        else:
            results['lstm'] = None
        
        # Prepare features for other models
        if features is None and sequences is not None:
            features = sequences.reshape(sequences.shape[0], -1)
        
        if features is not None and len(features) > 0:
            # Model 2: Isolation Forest
            try:
                isolation_pred = self.isolation_forest.predict(features)
                isolation_scores = self.isolation_forest.score_samples(features)
                # Convert: -1 (anomaly) -> True, 1 (normal) -> False
                isolation_flags = isolation_pred == -1
                # Normalize scores to [0, 1] (lower score = more anomalous)
                isolation_conf = 1.0 - (isolation_scores - isolation_scores.min()) / (
                    isolation_scores.max() - isolation_scores.min() + 1e-10
                )
                
                results['isolation'] = {
                    'anomaly_flags': isolation_flags,
                    'confidence_scores': isolation_conf,
                    'anomaly_rate': np.mean(isolation_flags)
                }
            except Exception as e:
                print(f"⚠️ Isolation Forest error: {e}")
                results['isolation'] = None
            
            # Model 3: Statistical Z-score
            try:
                z_scores = np.abs(
                    (features - self.statistical_mean) / (self.statistical_std + 1e-10)
                )
                max_z_scores = np.max(z_scores, axis=1)
                statistical_flags = max_z_scores > self.z_threshold
                statistical_conf = np.clip(max_z_scores / (self.z_threshold * 2), 0.0, 1.0)
                
                results['statistical'] = {
                    'anomaly_flags': statistical_flags,
                    'confidence_scores': statistical_conf,
                    'anomaly_rate': np.mean(statistical_flags)
                }
            except Exception as e:
                print(f"⚠️ Statistical detection error: {e}")
                results['statistical'] = None
            
            # Model 4: LOF
            try:
                lof_pred = self.lof.predict(features)
                lof_scores = -self.lof.score_samples(features)  # Negative for anomaly
                lof_flags = lof_pred == -1
                # Normalize scores
                lof_conf = (lof_scores - lof_scores.min()) / (
                    lof_scores.max() - lof_scores.min() + 1e-10
                )
                
                results['density'] = {
                    'anomaly_flags': lof_flags,
                    'confidence_scores': lof_conf,
                    'anomaly_rate': np.mean(lof_flags)
                }
            except Exception as e:
                print(f"⚠️ LOF detection error: {e}")
                results['density'] = None
        
        # Ensemble Fusion
        ensemble_result = self._fuse_ensemble_results(results)
        
        return {
            'individual_models': results,
            'ensemble': ensemble_result,
            'fusion_weights': self.weights
        }
    
    def _fuse_ensemble_results(self, individual_results: Dict) -> Dict:
        """
        Fuse results from individual models using majority voting
        Requires ≥2 of 4 detectors to agree for anomaly detection
        
        Args:
            individual_results: Results from each model
        
        Returns:
            Fused ensemble result
        """
        model_names = ['lstm', 'isolation', 'statistical', 'density']
        valid_models = []
        valid_flags = []
        valid_scores = []
        valid_weights = []
        
        for model_name in model_names:
            if individual_results.get(model_name) is not None:
                model_result = individual_results[model_name]
                valid_models.append(model_name)
                valid_flags.append(model_result['anomaly_flags'])
                valid_scores.append(model_result['confidence_scores'])
                valid_weights.append(self.weights[model_name])
        
        if len(valid_models) == 0:
            return {
                'anomaly_flags': np.array([]),
                'confidence_scores': np.array([]),
                'anomaly_rate': 0.0
            }
        
        n_samples = len(valid_flags[0])
        
        # Majority voting: ≥2 of 4 detectors must agree
        # Count votes for each sample
        vote_counts = np.zeros(n_samples, dtype=int)
        for flags in valid_flags:
            vote_counts += flags.astype(int)
        
        # Require at least 2 detectors to flag anomaly (majority of 4)
        # If we have fewer than 4 models, require at least half
        min_votes_required = max(2, len(valid_models) // 2 + 1)
        ensemble_flags = vote_counts >= min_votes_required
        
        # Weighted average for confidence scores (for calibration)
        total_weight = sum(valid_weights)
        normalized_weights = [w / total_weight for w in valid_weights]
        
        ensemble_scores = np.zeros(n_samples)
        for scores, weight in zip(valid_scores, normalized_weights):
            ensemble_scores += scores * weight
        
        return {
            'anomaly_flags': ensemble_flags,
            'confidence_scores': ensemble_scores,
            'anomaly_rate': np.mean(ensemble_flags),
            'contributing_models': valid_models,
            'vote_counts': vote_counts.tolist()  # For debugging
        }
    
    def adapt_weights(self, performance_history: Dict):
        """
        Adapt ensemble weights based on performance
        
        Args:
            performance_history: Dictionary with model performance metrics
        """
        # Simple adaptation: increase weight of better performing models
        # This is a placeholder - real adaptation would use more sophisticated methods
        total_performance = sum(performance_history.values())
        if total_performance > 0:
            for model_name in self.weights:
                if model_name in performance_history:
                    performance_ratio = performance_history[model_name] / total_performance
                    # Update weight (with smoothing)
                    self.weights[model_name] = (
                        0.7 * self.weights[model_name] + 0.3 * performance_ratio
                    )
            
            # Renormalize
            total_weight = sum(self.weights.values())
            for model_name in self.weights:
                self.weights[model_name] /= total_weight


if __name__ == "__main__":
    # Example usage
    ensemble = EnsembleAnomalyDetector()
    
    # Create dummy training data
    normal_sequences = np.random.rand(100, 60, 3)
    normal_features = normal_sequences.reshape(100, -1)
    
    # Train
    ensemble.train_ensemble(normal_sequences, normal_features)
    
    # Detect
    test_sequences = np.random.rand(10, 60, 3)
    test_features = test_sequences.reshape(10, -1)
    
    results = ensemble.detect_anomaly_ensemble(test_sequences, test_features)
    print(f"Ensemble anomaly rate: {results['ensemble']['anomaly_rate']:.2%}")
