"""
Real-Time Streaming Processing Layer

Handles continuous sensor streams with sliding-window analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import time
from datetime import datetime
import threading
import queue
import yaml


class StreamingProcessor:
    """
    Real-time streaming processor for continuous sensor data
    """
    
    def __init__(self, config_path: str = "config.yaml", window_size: int = 60):
        """Initialize streaming processor"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.window_size = window_size
        self.sequence_length = self.config['genai']['sequence_length']
        
        # Sliding window buffer
        self.buffer: Deque[Dict] = deque(maxlen=window_size)
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        
        # Detection latency tracking
        self.detection_times: List[float] = []
        self.anomaly_timestamps: List[float] = []
        
        # Real-time statistics
        self.stats = {
            'total_samples': 0,
            'anomalies_detected': 0,
            'avg_detection_latency': 0.0,
            'false_positives': 0,
            'true_positives': 0,
            'false_negatives': 0
        }
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def add_sample(self, timestamp: float, sensor_data: Dict) -> bool:
        """
        Add new sample to streaming buffer
        
        Args:
            timestamp: Current timestamp
            sensor_data: Dictionary with sensor readings
        
        Returns:
            True if buffer is ready for analysis
        """
        with self.lock:
            self.buffer.append(sensor_data)
            self.timestamps.append(timestamp)
            self.stats['total_samples'] += 1
            
            # Check if buffer is full enough for analysis
            return len(self.buffer) >= self.sequence_length
    
    def get_window_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current sliding window sequence for analysis
        
        Returns:
            (sequences, timestamps) - Ready for model inference
        """
        with self.lock:
            if len(self.buffer) < self.sequence_length:
                return None, None
            
            # Extract features
            features = []
            for sample in list(self.buffer)[-self.sequence_length:]:
                feature_vector = [
                    sample.get('level', 0.0),
                    sample.get('valve', 0),
                    sample.get('pump', 0)
                ]
                features.append(feature_vector)
            
            sequences = np.array([features])  # Shape: (1, sequence_length, n_features)
            timestamps = np.array(list(self.timestamps)[-self.sequence_length:])
            
            return sequences, timestamps
    
    def record_detection(self, timestamp: float, is_anomaly: bool, 
                        detection_latency: float = 0.0):
        """
        Record detection event for latency tracking
        
        Args:
            timestamp: Detection timestamp
            is_anomaly: Whether anomaly was detected
            detection_latency: Time from attack start to detection (seconds)
        """
        with self.lock:
            if is_anomaly:
                self.anomaly_timestamps.append(timestamp)
                self.detection_times.append(detection_latency)
                self.stats['anomalies_detected'] += 1
                
                # Update average latency
                if len(self.detection_times) > 0:
                    self.stats['avg_detection_latency'] = np.mean(self.detection_times)
    
    def update_ground_truth(self, is_anomaly: bool, detected: bool):
        """
        Update ground truth statistics
        
        Args:
            is_anomaly: True if attack actually occurred (ground truth)
            detected: True if system detected it
        """
        with self.lock:
            if is_anomaly and detected:
                self.stats['true_positives'] += 1
            elif is_anomaly and not detected:
                self.stats['false_negatives'] += 1
            elif not is_anomaly and detected:
                self.stats['false_positives'] += 1
    
    def get_statistics(self) -> Dict:
        """Get current streaming statistics"""
        with self.lock:
            return self.stats.copy()
    
    def reset(self):
        """Reset streaming processor"""
        with self.lock:
            self.buffer.clear()
            self.timestamps.clear()
            self.detection_times.clear()
            self.anomaly_timestamps.clear()
            self.stats = {
                'total_samples': 0,
                'anomalies_detected': 0,
                'avg_detection_latency': 0.0,
                'false_positives': 0,
                'true_positives': 0,
                'false_negatives': 0
            }


class RealTimeDetector:
    """
    Real-time anomaly detector with continuous operation
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize real-time detector"""
        self.config_path = config_path
        self.streaming_processor = StreamingProcessor(config_path)
        self.is_running = False
        self.detection_thread = None
        
    def start(self):
        """Start real-time detection loop"""
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
    
    def stop(self):
        """Stop real-time detection"""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5.0)
    
    def _detection_loop(self):
        """Internal detection loop (runs in separate thread)"""
        while self.is_running:
            if self.streaming_processor.get_window_sequence()[0] is not None:
                # Process window (would call ensemble models here)
                # This is a placeholder - actual detection happens in main system
                time.sleep(0.1)  # 10 Hz processing rate
            else:
                time.sleep(0.5)  # Wait for buffer to fill
    
    def process_stream_sample(self, timestamp: float, sensor_data: Dict,
                             ground_truth: Optional[bool] = None) -> Dict:
        """
        Process single stream sample
        
        Args:
            timestamp: Sample timestamp
            sensor_data: Sensor readings
            ground_truth: Optional ground truth label
        
        Returns:
            Detection result dictionary
        """
        # Add to buffer
        ready = self.streaming_processor.add_sample(timestamp, sensor_data)
        
        result = {
            'timestamp': timestamp,
            'ready_for_analysis': ready,
            'buffer_size': len(self.streaming_processor.buffer)
        }
        
        if ready:
            sequences, timestamps = self.streaming_processor.get_window_sequence()
            result['sequences_ready'] = True
            result['sequence_shape'] = sequences.shape if sequences is not None else None
        
        return result


if __name__ == "__main__":
    # Example usage
    processor = StreamingProcessor(window_size=60)
    
    # Simulate streaming data
    for t in range(100):
        sample = {
            'level': 500.0 + np.random.randn() * 5,
            'valve': 1,
            'pump': 0
        }
        ready = processor.add_sample(float(t), sample)
        if ready:
            seq, ts = processor.get_window_sequence()
            print(f"t={t}: Buffer ready, sequence shape={seq.shape}")
