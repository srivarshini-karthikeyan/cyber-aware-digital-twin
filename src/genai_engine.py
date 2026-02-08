"""
GenAI Layer: LSTM Autoencoder for Anomaly Detection and Attack Generation

Functions:
1. Learn normal behavior (unsupervised)
2. Generate synthetic anomaly sequences
3. Predict next-state deviation
4. Output anomaly confidence score
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Dict, List, Optional
import yaml
import pickle
from pathlib import Path


class LSTMAutoencoder:
    """
    LSTM Autoencoder for learning normal behavior and detecting anomalies
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize GenAI model with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        genai_config = self.config['genai']
        self.sequence_length = genai_config['sequence_length']
        self.latent_dim = genai_config['latent_dim']
        self.hidden_dims = genai_config['hidden_dims']
        self.learning_rate = genai_config['learning_rate']
        self.batch_size = genai_config['batch_size']
        self.epochs = genai_config['epochs']
        self.anomaly_threshold = genai_config['anomaly_threshold']
        
        self.model = None
        self.encoder = None
        self.decoder = None
        self.is_trained = False
        self.trained = False

        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM Autoencoder architecture
        
        Args:
            input_shape: (sequence_length, n_features)
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Encoder
        x = inputs
        for hidden_dim in self.hidden_dims:
            x = layers.LSTM(hidden_dim, return_sequences=True)(x)
            x = layers.Dropout(0.2)(x)
        
        # Latent representation
        encoded = layers.LSTM(self.latent_dim, return_sequences=False)(x)
        
        # Decoder
        x = layers.RepeatVector(self.sequence_length)(encoded)
        for hidden_dim in reversed(self.hidden_dims):
            x = layers.LSTM(hidden_dim, return_sequences=True)(x)
            x = layers.Dropout(0.2)(x)
        
        # Output layer
        decoded = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
        
        # Create model
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Create encoder and decoder separately
        self.encoder = keras.Model(inputs, encoded)
        
        decoder_input = keras.Input(shape=(self.latent_dim,))
        decoder_x = layers.RepeatVector(self.sequence_length)(decoder_input)
        for hidden_dim in reversed(self.hidden_dims):
            decoder_x = layers.LSTM(hidden_dim, return_sequences=True)(decoder_x)
        decoder_output = layers.TimeDistributed(layers.Dense(input_shape[1]))(decoder_x)
        self.decoder = keras.Model(decoder_input, decoder_output)
        
        self.model = autoencoder
        return autoencoder
    
    def train(self, train_sequences: np.ndarray, 
              val_sequences: Optional[np.ndarray] = None,
              verbose: int = 1) -> Dict:
        """
        Train the autoencoder on normal sequences
        
        Args:
            train_sequences: Training sequences (n_samples, sequence_length, n_features)
            val_sequences: Validation sequences (optional)
            verbose: Verbosity level
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            input_shape = (train_sequences.shape[1], train_sequences.shape[2])
            self.build_model(input_shape)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_sequences is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        validation_data = (val_sequences, val_sequences) if val_sequences is not None else None
        
        history = self.model.fit(
            train_sequences, train_sequences,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.is_trained = True
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', []),
            'mae': history.history.get('mae', []),
            'val_mae': history.history.get('val_mae', [])
        }
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Reconstruct sequences
        
        Args:
            sequences: Input sequences (n_samples, sequence_length, n_features)
        
        Returns:
            Reconstructed sequences
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(sequences, verbose=0)
    
    def compute_reconstruction_error(self, sequences: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection
        
        Args:
            sequences: Input sequences
        
        Returns:
            Reconstruction errors (MSE per sequence)
        """
        reconstructed = self.predict(sequences)
        
        # Compute MSE for each sequence
        errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
        
        return errors
    
    def detect_anomalies(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on reconstruction error
        
        Args:
            sequences: Input sequences to evaluate
        
        Returns:
            Tuple of (anomaly_flags, confidence_scores)
        """                 

        # 🔐 SAFETY GUARD: auto-train if model not trained
        if not self.trained:
            print("⚠️ GenAI model not trained. Training on baseline behavior...")
            self.train(sequences)
            print("✅ GenAI model training completed")

        # Compute reconstruction error
        errors = self.compute_reconstruction_error(sequences)

        # Normalize errors to [0, 1] for confidence scores
        max_error = np.max(errors) if np.max(errors) > 0 else 1.0
        confidence_scores = np.clip(errors / max_error, 0.0, 1.0)

        # Anomaly flags
        anomaly_flags = errors > self.anomaly_threshold

        return anomaly_flags, confidence_scores

    
    def generate_synthetic_anomaly(self, normal_sequence: np.ndarray,
                                   attack_type: str = "sensor_spoofing",
                                   intensity: float = 0.3) -> np.ndarray:
        """
        Generate synthetic anomaly sequence
        
        Args:
            normal_sequence: Normal sequence to perturb (1, sequence_length, n_features)
            attack_type: Type of attack to simulate
            intensity: Attack intensity [0, 1]
        
        Returns:
            Anomalous sequence
        """
        if len(normal_sequence.shape) == 2:
            normal_sequence = normal_sequence[np.newaxis, :, :]
        
        anomalous = normal_sequence.copy()
        
        if attack_type == "sensor_spoofing":
            # Spoof level sensor (first feature)
            anomalous[0, :, 0] = normal_sequence[0, :, 0] * (1.0 - intensity)
            
        elif attack_type == "slow_manipulation":
            # Gradual drift
            drift = np.linspace(0, intensity, normal_sequence.shape[1])
            anomalous[0, :, 0] = normal_sequence[0, :, 0] * (1.0 - drift)
            
        elif attack_type == "frozen_sensor":
            # Freeze at middle value
            mid_idx = normal_sequence.shape[1] // 2
            frozen_value = normal_sequence[0, mid_idx, 0]
            anomalous[0, :, 0] = frozen_value
            
        elif attack_type == "delayed_response":
            # Delay sensor response
            delay = int(normal_sequence.shape[1] * intensity)
            if delay > 0:
                anomalous[0, delay:, 0] = normal_sequence[0, :-delay, 0]
                anomalous[0, :delay, 0] = normal_sequence[0, 0, 0]
        
        elif attack_type == "noise_injection":
            # Add noise
            noise = np.random.normal(0, intensity, normal_sequence.shape[1])
            anomalous[0, :, 0] = normal_sequence[0, :, 0] + noise
        
        else:
            # Default: random perturbation
            noise = np.random.normal(0, intensity, normal_sequence.shape)
            anomalous = normal_sequence + noise
        
        return anomalous
    
    def generate_multiple_anomalies(self, normal_sequences: np.ndarray,
                                    num_anomalies: int = 10,
                                    attack_types: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate multiple synthetic anomaly sequences
        
        Args:
            normal_sequences: Normal sequences to perturb
            num_anomalies: Number of anomalies to generate
            attack_types: List of attack types (if None, uses config)
        
        Returns:
            Anomalous sequences
        """
        if attack_types is None:
            attack_types = self.config['genai']['attack_generation']['attack_types']
        
        anomalous_sequences = []
        
        for i in range(num_anomalies):
            # Randomly select a normal sequence
            idx = np.random.randint(0, len(normal_sequences))
            normal_seq = normal_sequences[idx:idx+1]
            
            # Randomly select attack type
            attack_type = np.random.choice(attack_types)
            intensity = np.random.uniform(0.2, 0.5)
            
            # Generate anomaly
            anomalous = self.generate_synthetic_anomaly(
                normal_seq, attack_type, intensity
            )
            anomalous_sequences.append(anomalous)
        
        return np.concatenate(anomalous_sequences, axis=0)
    
    def predict_next_state_deviation(self, current_sequence: np.ndarray) -> float:
        """
        Predict deviation of next state from normal behavior
        
        Args:
            current_sequence: Current sequence (1, sequence_length, n_features)
        
        Returns:
            Predicted deviation score [0, 1]
        """
        if len(current_sequence.shape) == 2:
            current_sequence = current_sequence[np.newaxis, :, :]
        
        # Reconstruct
        reconstructed = self.predict(current_sequence)
        
        # Compute deviation for last time step
        last_step_deviation = np.mean(
            np.abs(current_sequence[0, -1, :] - reconstructed[0, -1, :])
        )
        
        # Normalize
        deviation_score = np.clip(last_step_deviation, 0.0, 1.0)
        
        return float(deviation_score)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        
        # Save encoder and decoder
        encoder_path = filepath.replace('.h5', '_encoder.h5')
        decoder_path = filepath.replace('.h5', '_decoder.h5')
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Load encoder and decoder
        encoder_path = filepath.replace('.h5', '_encoder.h5')
        decoder_path = filepath.replace('.h5', '_decoder.h5')
        
        if Path(encoder_path).exists():
            self.encoder = keras.models.load_model(encoder_path)
        if Path(decoder_path).exists():
            self.decoder = keras.models.load_model(decoder_path)
        
        self.is_trained = True


if __name__ == "__main__":
    # Example usage
    genai = LSTMAutoencoder()
    
    # Create dummy data
    dummy_sequences = np.random.rand(100, 60, 3)  # 100 samples, 60 timesteps, 3 features
    
    # Build model
    genai.build_model((60, 3))
    print("Model built successfully")
    
    # Train (would use real data in practice)
    # history = genai.train(dummy_sequences[:70], dummy_sequences[70:85])
    
    # Generate anomaly
    # anomaly = genai.generate_synthetic_anomaly(dummy_sequences[0:1], "sensor_spoofing", 0.3)
    # print(f"Generated anomaly shape: {anomaly.shape}")
