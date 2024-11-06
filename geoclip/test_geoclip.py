"""
GeoCLIP Test Suite
=================

Comprehensive tests for GeoCLIP functionality.
"""

import unittest
import torch
import numpy as np
from geoclip.model import GeoCLIP
from geoclip.model.misc import convert_coordinates

class TestGeoCLIP(unittest.TestCase):
    def setUp(self):
        self.model = GeoCLIP(from_pretrained=True)
        self.test_coords = torch.tensor([[37.7749, -122.4194]])  # San Francisco

    def test_coordinate_conversion(self):
        """Test coordinate conversion utilities"""
        decimal = self.test_coords
        utm = convert_coordinates(decimal, 'decimal', 'utm')
        back_to_decimal = convert_coordinates(utm, 'utm', 'decimal')
        np.testing.assert_array_almost_equal(decimal.numpy(), back_to_decimal.numpy(), decimal=4)

    def test_model_prediction(self):
        """Test basic model prediction functionality"""
        with torch.no_grad():
            logits = self.model(
                torch.randn(1, 3, 224, 224),  # Dummy image
                self.test_coords
            )
        self.assertEqual(logits.shape, (1, 1))

    def test_queue_mechanism(self):
        """Test GPS queue updates"""
        initial_queue = self.model.get_gps_queue().clone()
        self.model.dequeue_and_enqueue(self.test_coords)
        updated_queue = self.model.get_gps_queue()
        self.assertFalse(torch.equal(initial_queue, updated_queue))

if __name__ == '__main__':
    unittest.main()
```

2. Logging System:

```python
"""
GeoCLIP Logging Utilities
========================

Structured logging system for training and inference.
"""

import logging
import json
from pathlib import Path
from datetime import datetime

class GeoCLIPLogger:
    def __init__(self, log_dir="logs", experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup experiment
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Metrics tracking
        self.metrics = {
            'training_loss': [],
            'validation_metrics': [],
            'queue_statistics': [],
            'coordinate_coverage': {}
        }
    
    def setup_logging(self):
        """Configure logging handlers"""
        log_file = self.experiment_dir / "experiment.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("GeoCLIP")
    
    def log_training_step(self, epoch, batch, loss, learning_rate):
        """Log training step metrics"""
        self.metrics['training_loss'].append({
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'lr': learning_rate
        })
        
        self.logger.info(
            f"Epoch {epoch}, Batch {batch}: Loss={loss:.4f}, LR={learning_rate:.6f}"
        )
    
    def log_validation_metrics(self, metrics):
        """Log validation results"""
        self.metrics['validation_metrics'].append(metrics)
        
        self.logger.info(
            f"Validation Results:\n"
            f"Median Distance: {metrics['median_distance']:.2f}km\n"
            f"Mean Distance: {metrics['mean_distance']:.2f}km\n"
            f"90th Percentile: {metrics['percentile_90']:.2f}km"
        )
    
    def log_queue_stats(self, queue_age, queue_usage):
        """Log GPS queue statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'mean_age': float(queue_age.mean()),
            'max_age': float(queue_age.max()),
            'usage_distribution': queue_usage.tolist()
        }
        self.metrics['queue_statistics'].append(stats)
    
    def save_experiment(self):
        """Save all experiment metrics"""
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Experiment metrics saved to {metrics_file}")
```

3. Data Validation:

```python
"""
GeoCLIP Data Validation
======================

Utilities for validating input data and model outputs.
"""

import torch
import pandas as pd
from typing import Union, Tuple, List
from pathlib import Path

class DataValidator:
    """Validates input data for GeoCLIP"""
    
    @staticmethod
    def validate_coordinates(
        coords: Union[torch.Tensor, pd.DataFrame],
        allow_duplicates: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate coordinate data format and values.
        
        Args:
            coords: Coordinate data
            allow_duplicates: Whether to allow duplicate coordinates
        
        Returns:
            (is_valid, message)
        """
        if isinstance(coords, pd.DataFrame):
            # Check column names
            if not all(col in coords.columns for col in ['LAT', 'LON']):
                return False, "Missing required columns 'LAT' or 'LON'"
            
            # Convert to tensor for validation
            coords = torch.tensor(coords[['LAT', 'LON']].values)
        
        # Check shape and types
        if coords.dim() != 2 or coords.shape[1] != 2:
            return False, "Coordinates must be shape (N, 2)"
        
        # Validate coordinate ranges
        if torch.any(coords[:, 0] < -90) or torch.any(coords[:, 0] > 90):
            return False, "Latitude must be between -90 and 90"
        if torch.any(coords[:, 1] < -180) or torch.any(coords[:, 1] > 180):
            return False, "Longitude must be between -180 and 180"
        
        # Check duplicates
        if not allow_duplicates:
            if len(coords) != len(torch.unique(coords, dim=0)):
                return False, "Duplicate coordinates found"
        
        return True, "Valid coordinates"
    
    @staticmethod
    def validate_image_paths(paths: List[str]) -> Tuple[bool, str, List[str]]:
        """
        Validate image file paths.
        
        Returns:
            (is_valid, message, invalid_paths)
        """
        invalid_paths = []
        supported_extensions = {'.jpg', '.jpeg', '.png'}
        
        for path in paths:
            p = Path(path)
            if not p.exists():
                invalid_paths.append(f"{path} (not found)")
            elif p.suffix.lower() not in supported_extensions:
                invalid_paths.append(f"{path} (unsupported format)")
            
        if invalid_paths:
            return False, "Invalid image paths found", invalid_paths
        return True, "All paths valid", []
    
    @staticmethod
    def validate_batch_size(batch_size: int, queue_size: int) -> bool:
        """Validate batch size compatibility with queue size"""
        return queue_size % batch_size == 0
    
    @staticmethod
    def validate_model_output(
        logits: torch.Tensor,
        batch_size: int,
        num_locations: int
    ) -> bool:
        """Validate model output dimensions"""
        expected_shape = (batch_size, num_locations)
        return logits.shape == expected_shape
```

4. Visualization Tools:

```python
"""
GeoCLIP Visualization Utilities
==============================

Tools for visualizing model predictions and training progress.
"""

import matplotlib.pyplot as plt
import folium
from typing import List, Tuple, Dict
import torch
import numpy as np

class GeoCLIPVisualizer:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_training_progress(
        self,
        metrics: Dict[str, List[float]],
        save_name: str = "training_progress.png"
    ):
        """Plot training metrics over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot loss
        epochs = range(1, len(metrics['training_loss']) + 1)
        ax1.plot(epochs, metrics['training_loss'], 'b-')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        # Plot validation metrics
        if metrics['validation_metrics']:
            val_epochs = range(1, len(metrics['validation_metrics']) + 1)
            distances = [m['median_distance'] for m in metrics['validation_metrics']]
            ax2.plot(val_epochs, distances, 'r-')
            ax2.set_title('Median Distance Error')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Distance (km)')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def visualize_predictions(
        self,
        image_path: str,
        true_coords: Tuple[float, float],
        pred_coords: List[Tuple[float, float]],
        confidences: List[float],
        save_name: str = "prediction_map.html"
    ):
        """Create interactive map of predictions"""
        # Center map on true location
        m = folium.Map(
            location=true_coords,
            zoom_start=4
        )
        
        # Add true location
        folium.Marker(
            true_coords,
            popup="True Location",
            icon=folium.Icon(color='green')
        ).add_to(m)
        
        # Add predicted locations
        for (lat, lon), conf in zip(pred_coords, confidences):
            folium.Marker(
                [lat, lon],
                popup=f"Confidence: {conf:.2%}",
                icon=folium.Icon(color='red', opacity=conf)
            ).add_to(m)
        
        # Save map
        m.save(self.save_dir / save_name)
    
    def plot_coordinate_coverage(
        self,
        coordinates: torch.Tensor,
        density: bool = True,
        save_name: str = "coordinate_coverage.png"
    ):
        """Plot geographic coverage of coordinate gallery"""
        plt.figure(figsize=(15, 10))
        
        if density:
            plt.hist2d(
                coordinates[:, 1].numpy(),  # longitude
                coordinates[:, 0].numpy(),  # latitude
                bins=50,
                cmap='viridis'
            )
            plt.colorbar(label='Density')
        else:
            plt.scatter(
                coordinates[:, 1].numpy(),
                coordinates[:, 0].numpy(),
                alpha=0.5,
                s=1
            )
        
        plt.title('Coordinate Gallery Coverage')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        
        plt.savefig(self.save_dir / save_name)
        plt.close()
