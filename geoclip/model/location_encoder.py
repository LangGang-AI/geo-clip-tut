"""
Location Encoder Neural Network Module
====================================

This module implements a sophisticated location encoding system using the Equal Earth map projection
and multi-scale Random Fourier Features (RFF) for geographic coordinate embedding.

Key Components:
- Equal Earth Projection: A novel equal-area pseudocylindrical projection for accurate geographic representation
- Multi-scale Location Encoder: Processes locations at different spatial scales using multiple encoder capsules
- RFF-based Encoding: Utilizes Random Fourier Features for effective coordinate embedding

Technical Details:
- Input: Geographic coordinates (latitude, longitude) in degrees
- Output: 512-dimensional location feature embeddings
- Projection Scale Factor: 66.50336 (SF constant)
- Supported sigma values: Default scales at 2^0, 2^4, and 2^8 kilometers

Author: Unknown
License: Unknown
"""

import torch
import torch.nn as nn
from .rff import GaussianEncoding
from .misc import file_dir

# Equal Earth projection constants
# These coefficients are derived from the projection's mathematical formulation
# See: https://doi.org/10.1080/13658816.2018.1504949
A1 = 1.340264    # Primary scale factor
A2 = -0.081106   # Second-order correction
A3 = 0.000893    # Third-order correction
A4 = 0.003796    # Fourth-order correction
SF = 66.50336    # Final scale factor for coordinate normalization

def equal_earth_projection(L):
    """
    Transforms geographic coordinates to Equal Earth projection coordinates.
    
    Args:
        L (torch.Tensor): Batch of [latitude, longitude] pairs in degrees
            Shape: (batch_size, 2)
    
    Returns:
        torch.Tensor: Projected coordinates normalized by scale factor
            Shape: (batch_size, 2)
    """
    latitude = L[:, 0]
    longitude = L[:, 1]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    
    # Calculate auxiliary theta parameter
    sin_theta = (torch.sqrt(torch.tensor(3.0)) / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    
    # Calculate projection denominator using polynomial terms
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    
    # Calculate x (projected longitude) and y (projected latitude)
    x = (2 * torch.sqrt(torch.tensor(3.0)) * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    
    return (torch.stack((x, y), dim=1) * SF) / 180

class LocationEncoderCapsule(nn.Module):
    """
    Single-scale location encoder capsule using RFF encoding and MLP.
    
    Each capsule processes locations at a specific spatial scale (sigma)
    through a series of linear transformations and ReLU activations.
    
    Args:
        sigma (float): Spatial scale parameter in kilometers for RFF encoding
    """
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        
        # Main processing pipeline
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # Final dimension reduction head
        self.head = nn.Sequential(nn.Linear(1024, 512))
    
    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class LocationEncoder(nn.Module):
    """
    Multi-scale location encoder combining multiple capsules at different spatial scales.
    
    Creates a rich location embedding by combining features from multiple spatial scales,
    allowing the model to capture both local and global geographic relationships.
    
    Args:
        sigma (list): List of spatial scales in kilometers for the encoder capsules
            Default: [2^0, 2^4, 2^8] = [1km, 16km, 256km]
        from_pretrained (bool): Whether to load pre-trained weights
            Default: True
    """
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=True):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        
        # Create encoder capsules for each spatial scale
        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))
            
        if from_pretrained:
            self._load_weights()
    
    def _load_weights(self):
        """Loads pre-trained weights from the specified file path."""
        self.load_state_dict(torch.load(f"{file_dir}/weights/location_encoder_weights.pth"))
    
    def forward(self, location):
        """
        Processes geographic coordinates through multiple spatial scales.
        
        Args:
            location (torch.Tensor): Batch of [latitude, longitude] coordinates
                Shape: (batch_size, 2)
        
        Returns:
            torch.Tensor: Combined location features
                Shape: (batch_size, 512)
        """
        # Project geographic coordinates to Equal Earth projection
        location = equal_earth_projection(location)
        
        # Initialize output tensor
        location_features = torch.zeros(location.shape[0], 512).to(location.device)
        
        # Aggregate features from all spatial scales
        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location)
        
        return location_features
