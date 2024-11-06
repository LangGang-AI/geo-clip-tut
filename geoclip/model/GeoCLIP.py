"""
GeoCLIP: Geographic Location Prediction from Images
================================================

This module implements GeoCLIP, a neural network that predicts geographic locations from images
by combining CLIP-based image features with learned geographic embeddings.

Architecture Overview:
- Image Branch: Modified CLIP vision encoder with custom MLP head
- Location Branch: Multi-scale geographic coordinate encoder
- Training Strategy: Contrastive learning with dynamic GPS queue
- Loss: InfoNCE-style contrastive loss with learned temperature parameter

Key Components:
- Image Encoder: CLIP-based visual feature extractor (→ 512D embeddings)
- Location Encoder: Multi-scale geographic coordinate embedder (→ 512D embeddings)
- GPS Queue: Dynamic memory bank for contrastive learning (default 4096 locations)
- Temperature: Learned scaling factor for similarity scores

Technical Details:
- Input: Images (224x224 RGB) and GPS coordinates (lat/lon pairs)
- Output: Similarity scores between image and location embeddings
- Pretrained weights available for all components
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    """
    GeoCLIP model for predicting geographic locations from images using
    contrastive learning between image and location embeddings.
    
    The model learns to align visual features with geographic embeddings
    in a shared semantic space, enabling image-to-location retrieval.
    
    Args:
        from_pretrained (bool): Whether to load pretrained weights (default: True)
        queue_size (int): Size of the GPS memory queue (default: 4096)
    """
    
    def __init__(self, from_pretrained=True, queue_size=4096):
        super().__init__()
        # Initialize temperature parameter for similarity scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Initialize encoders
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()
        
        # Load GPS gallery for inference
        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv"))
        self._initialize_gps_queue(queue_size)
        
        # Load pretrained weights if requested
        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()
        
        self.device = "cpu"

    def to(self, device):
        """
        Moves the model to specified device and ensures all buffers follow.
        
        Args:
            device: Target device (e.g., 'cuda', 'cpu')
        
        Returns:
            GeoCLIP: Self reference for method chaining
        """
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        """
        Loads pretrained weights for all model components from the weights folder.
        """
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))

    def _initialize_gps_queue(self, queue_size):
        """
        Initializes the GPS coordinate queue for contrastive learning.
        
        Args:
            queue_size (int): Number of GPS coordinates to maintain in queue
        """
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """
        Updates GPS queue with new coordinates using FIFO strategy.
        
        Args:
            gps (torch.Tensor): GPS coordinates to enqueue
                Shape: (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"
        
        # Replace oldest GPS coordinates with new batch
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        """
        Returns the current GPS queue coordinates.
        
        Returns:
            torch.Tensor: Queue GPS coordinates
                Shape: (queue_size, 2)
        """
        return self.gps_queue.t()
                                             
    def forward(self, image, location):
        """
        Computes similarity scores between image and location embeddings.
        
        Args:
            image (torch.Tensor): Batch of images
                Shape: (n, 3, 224, 224)
            location (torch.Tensor): Batch of GPS coordinates
                Shape: (m, 2)

        Returns:
            torch.Tensor: Similarity scores between each image-location pair
                Shape: (n, m)
        
        Processing Steps:
        1. Extract normalized image and location features
        2. Scale features by learned temperature
        3. Compute cosine similarities
        """
        # Compute and normalize features
        image_features = F.normalize(self.image_encoder(image), dim=1)
        location_features = F.normalize(self.location_encoder(location), dim=1)
        
        # Scale by learned temperature
        logit_scale = self.logit_scale.exp()
        
        # Compute similarities
        logits_per_image = logit_scale * (image_features @ location_features.t())
        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """
        Predicts most likely GPS coordinates for a given image.
        
        Args:
            image_path (str): Path to input image
            top_k (int): Number of top predictions to return

        Returns:
            tuple:
                - top_pred_gps (torch.Tensor): Top-k predicted coordinates
                    Shape: (k, 2)
                - top_pred_prob (torch.Tensor): Confidence scores for predictions
                    Shape: (k,)
        """
        # Load and preprocess image
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        # Move GPS gallery to same device
        gps_gallery = self.gps_gallery.to(self.device)

        # Compute similarities and probabilities
        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top-k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob
