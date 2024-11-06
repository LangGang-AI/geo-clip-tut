"""
GeoCLIP
=======
Simple interface for geographic embeddings, location prediction, and similarity comparison.

Basic Usage:
    from geoclip import get_embeddings, predict_location, compare_locations
    
    # Get embeddings
    locations = [[40.7128, -74.0060]]  # NYC
    embeddings = get_embeddings(locations)
    
    # Predict location
    predictions = predict_location("photo.jpg")
    print(predictions)
    
    # Compare locations
    locations = [[40.7128, -74.0060], [34.0522, -118.2437]]  # NYC and LA
    similarity = compare_locations(locations)
    print(similarity)
"""

from .types import (
    Embedding,
    EmbeddingType,
    LocationPrediction,
    PredictionSet,
    LocationComparison,
    LocationComparer,
    DistanceMetric
)

from .model import (
    GeoCLIP,
    ImageEncoder,
    LocationEncoder
)

from .train import train

# Define main API functions
def get_embeddings(locations: List[Tuple[float, float]]) -> List[Embedding]:
    """Get embeddings for locations"""
    return ModelFactory.get_encoder().encode(locations)

def predict_location(image_path: str, top_k: int = 5) -> PredictionSet:
    """Predict location from image"""
    return ModelFactory.get_predictor().predict(image_path, top_k)

def compare_locations(
    locations: Union[List[Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Union[LocationComparison, MultiLocationSimilarity]:
    """Compare locations using their geographic embeddings"""
    return ModelFactory.get_comparator().compare(locations)

# Define public API
__all__ = [
    # Main functions
    'get_embeddings',
    'predict_location',
    'compare_locations',
    
    # Types
    'Embedding',
    'EmbeddingType',
    'LocationPrediction',
    'PredictionSet',
    'LocationComparison',
    'LocationComparer',
    'DistanceMetric',
    
    # Models
    'GeoCLIP',
    'ImageEncoder',
    'LocationEncoder'
]
