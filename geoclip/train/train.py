"""
Training Function for GeoCLIP

This module implements a training function for neural networks that process paired image and GPS data.
It utilizes a queue-based mechanism for maintaining spatial context across batches, making it 
particularly effective for geospatial learning tasks.

Key Components:
    - GPS Queue System: Maintains historical geographic context
    - Loss Computation: Uses cross-entropy for spatial relationship learning
    - Batch Processing: Handles image and GPS data simultaneously
    - Progress Tracking: Real-time monitoring of training metrics

Memory Requirements:
    - Scales with batch size and GPS queue length
    - Requires memory for both image and GPS data storage
    - Additional memory needed for gradient computation
"""

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def train(train_dataloader, model, optimizer, epoch, batch_size, device, scheduler=None, criterion=nn.CrossEntropyLoss()):
    """
    Executes one epoch of training for geospatial deep learning.

    Args:
        train_dataloader (DataLoader): Provides batches of training data
            - Expected to yield (images, gps) tuples
            - Must maintain consistent batch size
            - Handles data shuffling internally

        model (nn.Module): Neural network model that must implement:
            - get_gps_queue(): Returns stored GPS coordinates
            - dequeue_and_enqueue(gps): Updates GPS queue
            - forward(imgs, gps_all): Processes combined data

        optimizer (torch.optim.Optimizer): Updates model weights
            - Compatible with any PyTorch optimizer
            - Handles gradient application
            - Manages parameter updates

        epoch (int): Current training epoch number
            - Used for progress tracking
            - Applied in scheduler steps
            - Aids in training monitoring

        batch_size (int): Size of each training batch
            - Must match DataLoader configuration
            - Used for target tensor creation
            - Affects memory requirements

        device (torch.device): Computation device
            - 'cuda' for GPU processing
            - 'cpu' for CPU processing
            - Handles tensor placement

        scheduler (optional): Learning rate scheduler
            - Adjusts learning rate per epoch
            - Applied at epoch end
            - Helps optimize convergence

        criterion (nn.Module, optional): Loss function
            - Defaults to CrossEntropyLoss
            - Must match model output format
            - Determines error calculation

    Training Process:
        1. Batch Processing:
           - Load image and GPS data
           - Transfer to computation device
           - Update GPS queue

        2. Forward Pass:
           - Process through model
           - Compute loss
           - Prepare for backpropagation

        3. Optimization:
           - Calculate gradients
           - Update model weights
           - Clear gradients for next batch

    Returns:
        None: Updates model in-place through optimizer

    Example:
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>> model = GeospatialNet()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> device = torch.device('cuda')
        >>> train(dataloader, model, optimizer, epoch=1, 
                 batch_size=32, device=device)
    """
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)

    for i, (imgs, gps) in bar:
        imgs = imgs.to(device)
        gps = gps.to(device)

        gps_queue = model.get_gps_queue()

        optimizer.zero_grad()

        gps_all = torch.cat([gps, gps_queue], dim=0)

        model.dequeue_and_enqueue(gps)

        logits_img_gps = model(imgs, gps_all)

        img_gps_loss = criterion(logits_img_gps, targets_img_gps)

        loss = img_gps_loss

        loss.backward()

        optimizer.step()

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    if scheduler is not None:
        scheduler.step()
