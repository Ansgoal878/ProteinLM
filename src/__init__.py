"""
ProteinLM: A simplified toolkit for protein sequence embedding generation and visualization
using pretrained language models from Hugging Face.
"""

from .model_loader import load_model, list_available_models
from .embedding_generator import ProteinEmbeddingGenerator
from .visualizer import EmbeddingVisualizer

__all__ = [
    'load_model',
    'list_available_models',
    'ProteinEmbeddingGenerator',
    'EmbeddingVisualizer'
]

__version__ = '0.1.0'