"""
Model Versioning & Persistence System

Manages model versions, experiment reproducibility, and persistent artifacts
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import yaml


@dataclass
class ModelVersion:
    """Model version information"""
    version: str
    timestamp: str
    model_type: str
    config_hash: str
    training_data_hash: Optional[str]
    performance_metrics: Dict
    file_path: str
    description: str


class ModelVersionManager:
    """
    Manages model versions and experiment reproducibility
    """
    
    def __init__(self, models_dir: str = "models/saved_models"):
        """Initialize version manager"""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.models_dir / "versions.json"
        self.versions: List[ModelVersion] = []
        self.load_versions()
    
    def save_model(self, model, model_type: str, config: Dict,
                  performance_metrics: Dict,
                  training_data_hash: Optional[str] = None,
                  description: str = "") -> str:
        """
        Save model with versioning
        
        Args:
            model: Model object to save
            model_type: Type of model (e.g., 'lstm_autoencoder', 'ensemble')
            config: Configuration dictionary
            performance_metrics: Performance metrics dictionary
            training_data_hash: Hash of training data (for reproducibility)
            description: Human-readable description
        
        Returns:
            Version string
        """
        # Generate version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{model_type}_{timestamp}"
        
        # Compute config hash
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Save model
        model_file = self.models_dir / f"{version}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Create version record
        version_record = ModelVersion(
            version=version,
            timestamp=timestamp,
            model_type=model_type,
            config_hash=config_hash,
            training_data_hash=training_data_hash,
            performance_metrics=performance_metrics,
            file_path=str(model_file),
            description=description
        )
        
        self.versions.append(version_record)
        self.save_versions()
        
        return version
    
    def load_model(self, version: str):
        """
        Load model by version
        
        Args:
            version: Version string
        
        Returns:
            Loaded model object
        """
        version_record = self.get_version(version)
        if version_record is None:
            raise ValueError(f"Version {version} not found")
        
        with open(version_record.file_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get version record"""
        for v in self.versions:
            if v.version == version:
                return v
        return None
    
    def get_latest_version(self, model_type: Optional[str] = None) -> Optional[ModelVersion]:
        """Get latest version (optionally filtered by type)"""
        filtered = self.versions
        if model_type:
            filtered = [v for v in self.versions if v.model_type == model_type]
        
        if not filtered:
            return None
        
        return max(filtered, key=lambda v: v.timestamp)
    
    def list_versions(self, model_type: Optional[str] = None) -> List[ModelVersion]:
        """List all versions (optionally filtered)"""
        if model_type:
            return [v for v in self.versions if v.model_type == model_type]
        return self.versions.copy()
    
    def save_versions(self):
        """Save versions to file"""
        versions_data = [asdict(v) for v in self.versions]
        with open(self.versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2)
    
    def load_versions(self):
        """Load versions from file"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
            self.versions = [ModelVersion(**v) for v in versions_data]
        else:
            self.versions = []
    
    def create_experiment_log(self, experiment_name: str, config: Dict,
                             results: Dict, output_dir: str = "outputs/experiments") -> str:
        """
        Create experiment log for reproducibility
        
        Args:
            experiment_name: Name of experiment
            config: Configuration used
            results: Experiment results
            output_dir: Output directory
        
        Returns:
            Path to experiment log
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_path / f"{experiment_name}_{timestamp}.json"
        
        experiment_log = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config': config,
            'results': results,
            'model_versions': [v.version for v in self.versions]
        }
        
        with open(log_file, 'w') as f:
            json.dump(experiment_log, f, indent=2, default=str)
        
        return str(log_file)


if __name__ == "__main__":
    # Example usage
    manager = ModelVersionManager()
    
    # Save a model
    dummy_model = {"weights": [1, 2, 3]}
    config = {"learning_rate": 0.001, "epochs": 50}
    metrics = {"accuracy": 0.95, "f1": 0.92}
    
    version = manager.save_model(
        dummy_model, "lstm_autoencoder", config, metrics,
        description="Initial training run"
    )
    
    print(f"Saved model version: {version}")
    
    # List versions
    versions = manager.list_versions()
    print(f"Total versions: {len(versions)}")
    
    # Get latest
    latest = manager.get_latest_version("lstm_autoencoder")
    if latest:
        print(f"Latest version: {latest.version}")
