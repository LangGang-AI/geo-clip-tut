"""Core type definitions for GeoCLIP"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, List, Optional, Protocol, runtime_checkable, Tuple, Dict
import numpy as np
from enum import Enum
import math
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingType(Enum):
    """Types of embeddings supported by the model"""
    IMAGE = "image"
    LOCATION = "location"
    JOINT = "joint"

class DistanceMetric(Enum):
    """Supported distance calculation methods"""
    HAVERSINE = "haversine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

@dataclass(frozen=True)
class Embedding:
    """Immutable embedding vector with metadata"""
    vector: np.ndarray
    type: EmbeddingType
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.vector, np.ndarray):
            object.__setattr__(self, 'vector', np.array(self.vector))
    
    def similarity_to(self, other: Embedding) -> float:
        """Calculate cosine similarity with another embedding"""
        return float(cosine_similarity(
            self.vector.reshape(1, -1),
            other.vector.reshape(1, -1)
        )[0, 0])

    def __len__(self) -> int:
        return len(self.vector)
    
    def __str__(self) -> str:
        return f"{self.type.value.title()} Embedding (dim={len(self)})"

# [Rest of the type definitions from previous code...]
