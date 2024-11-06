"""
Image Encoder
=======================

This module implements an image encoder that leverages OpenAI's CLIP (Contrastive Language-Image Pre-training) 
model to extract rich visual features, followed by dimensionality reduction through an MLP.

Architecture Overview:
- Base Model: CLIP ViT-Large/14 (Vision Transformer with 14x14 patch size)
- Feature Dimension: 768 → 512 (after MLP)
- Training Mode: CLIP weights frozen, only MLP is trainable

Key Components:
- CLIP Vision Encoder: Pre-trained vision transformer for robust image feature extraction
- Image Processor: Handles image preprocessing according to CLIP's requirements
- Dimension Reduction MLP: Transforms CLIP's 768D features to 512D embeddings

Technical Details:
- Input: Raw images (various formats supported by CLIP processor)
- Output: 512-dimensional image feature embeddings
- Model Source: 'openai/clip-vit-large-patch14'

Note: This implementation suppresses HuggingFace Hub warnings for cleaner output.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
import warnings

# Suppress HuggingFace Hub warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    """
    Neural network module that combines CLIP's vision encoder with a custom MLP
    for generating fixed-dimensional image embeddings.
    
    The model architecture:
    1. CLIP ViT processes the image to extract high-level features (768D)
    2. MLP reduces dimensionality while preserving semantic information (512D)
    
    Note: CLIP weights are frozen to preserve pre-trained knowledge while
    the MLP layers are trainable for task-specific adaptation.
    """
    
    def __init__(self):
        """
        Initializes the ImageEncoder with a pre-trained CLIP model and custom MLP.
        
        Components initialized:
        - CLIP model: Pre-trained vision transformer
        - Image processor: CLIP-specific image preprocessing
        - MLP: Dimensionality reduction network (768D → 512D)
        """
        super(ImageEncoder, self).__init__()
        
        # Initialize CLIP model and processor
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Define dimensionality reduction MLP
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),  # Preserve dimensionality initially
            nn.ReLU(),            # Non-linear activation
            nn.Linear(768, 512)   # Reduce to final dimension
        )
        
        # Freeze CLIP parameters to prevent fine-tuning
        for param in self.CLIP.parameters():
            param.requires_grad = False
    
    def preprocess_image(self, image):
        """
        Preprocesses input images according to CLIP's requirements.
        
        Args:
            image: Raw input image(s) in format supported by CLIP processor
                (PIL Image, numpy array, torch tensor, etc.)
        
        Returns:
            torch.Tensor: Preprocessed image tensor ready for CLIP
                Shape: (batch_size, channels, height, width)
        """
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x
    
    def forward(self, x):
        """
        Processes images through CLIP and MLP to generate embeddings.
        
        Args:
            x (torch.Tensor): Preprocessed image tensor from preprocess_image()
                Shape: (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Image embeddings
                Shape: (batch_size, 512)
        
        Processing Steps:
        1. Extract CLIP image features (768D)
        2. Transform through MLP to final dimensions (512D)
        """
        # Get CLIP image features
        x = self.CLIP.get_image_features(pixel_values=x)
        
        # Transform through MLP
        x = self.mlp(x)
        
        return x
