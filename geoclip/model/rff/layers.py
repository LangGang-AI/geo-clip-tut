"""
Random Fourier Features and Positional Encoding Layers
===================================================

This module implements neural network layers for coordinate encoding using:
1. Random Fourier Features (RFF) with Gaussian sampling
2. Basic trigonometric encoding
3. Multi-scale positional encoding

These encodings transform low-dimensional coordinates into higher-dimensional
feature spaces, making them more suitable for neural network processing.

Mathematical Background:
- RFF approximates shift-invariant kernels in feature space
- Positional encoding captures information at multiple frequency scales
- Basic encoding provides simple periodic feature mapping

Implementation Details:
- All encodings use 2π-periodic trigonometric functions
- GaussianEncoding uses randomly sampled projection matrices
- PositionalEncoding uses geometric progression of frequencies
"""

import torch.nn as nn
from typing import Optional
from torch import Tensor
from . import functional

class GaussianEncoding(nn.Module):
    """
    Neural network layer implementing Random Fourier Feature mapping with
    Gaussian-sampled projection matrices.
    
    This encoding transforms input coordinates using the formula:
    γ(v) = [cos(2πBv), sin(2πBv)]
    where B is a randomly sampled matrix with entries from N(0, σ²).
    
    Key Features:
    - Approximates shift-invariant kernels
    - Provides fixed-size output regardless of input dimension
    - Non-trainable random projections
    """
    
    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[float] = None,
                 encoded_size: Optional[float] = None,
                 b: Optional[Tensor] = None):
        """
        Initializes the Gaussian encoding layer.
        
        Args:
            sigma: Standard deviation for random matrix sampling
            input_size: Dimension of input coordinates
            encoded_size: Desired output feature dimension (before concatenation)
            b: Optional pre-sampled projection matrix
                Shape: (encoded_size, input_size)
        
        Note: Either provide (sigma, input_size, encoded_size) OR just b,
              but not both combinations.
        """
        super().__init__()
        
        # Handle initialization modes
        if b is None:
            # Initialize with new random matrix
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')
            b = functional.sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
            
        # Register projection matrix as non-trainable parameter
        self.b = nn.parameter.Parameter(b, requires_grad=False)
    
    def forward(self, v: Tensor) -> Tensor:
        """
        Applies Gaussian RFF encoding to input coordinates.
        
        Args:
            v: Input coordinates
                Shape: (batch_size, *, input_size)
        
        Returns:
            Encoded features
                Shape: (batch_size, *, 2 * encoded_size)
        """
        return functional.gaussian_encoding(v, self.b)

class BasicEncoding(nn.Module):
    """
    Simple periodic encoding layer that applies trigonometric functions
    directly to input coordinates.
    
    The encoding doubles the feature dimension by concatenating
    cos(2πv) and sin(2πv) for each input dimension.
    """
    
    def forward(self, v: Tensor) -> Tensor:
        """
        Applies basic trigonometric encoding to input coordinates.
        
        Args:
            v: Input coordinates
                Shape: (batch_size, *, input_size)
        
        Returns:
            Encoded features
                Shape: (batch_size, *, 2 * input_size)
        """
        return functional.basic_encoding(v)

class PositionalEncoding(nn.Module):
    """
    Multi-scale positional encoding layer using geometric progression
    of frequencies, similar to the encoding used in transformer models.
    
    The encoding captures patterns at different scales by using
    multiple frequency bands: σ^(j/m) for j in {0,...,m-1}
    """
    
    def __init__(self, sigma: float, m: int):
        """
        Initializes the positional encoding layer.
        
        Args:
            sigma: Base for geometric progression of frequencies
            m: Number of frequency bands to use
        """
        super().__init__()
        self.sigma = sigma  # Frequency base
        self.m = m         # Number of frequency bands
    
    def forward(self, v: Tensor) -> Tensor:
        """
        Applies multi-scale positional encoding to input coordinates.
        
        Args:
            v: Input coordinates
                Shape: (batch_size, *, input_size)
        
        Returns:
            Encoded features combining multiple frequency bands
                Shape: (batch_size, *, 2 * m * input_size)
        """
        return functional.positional_encoding(v, self.sigma, self.m)
