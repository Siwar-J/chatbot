# src/config.py
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration optimisée pour 4GB GPU"""
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Modèles légers compatibles 4GB GPU
    LLM_MODEL: str = "microsoft/DialoGPT-medium"  # ~800MB - Parfait pour 4GB
    
    # Paramètres optimisés pour petits modèles
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.1

@dataclass
class ProcessingConfig:
    """Configuration du traitement"""
    CHUNK_SIZE: int = 600  # Réduit pour petits modèles
    CHUNK_OVERLAP: int = 80
    SIMILARITY_TOP_K: int = 3  # Réduit pour économiser la mémoire

@dataclass
class PathConfig:
    """Configuration des chemins"""
    UPLOAD_DIR: str = "data/uploaded_docs"
    VECTOR_STORE_DIR: str = "data/vector_stores"
    STATIC_DIR: str = "static"
    
    def create_directories(self):
        """CORRECTION: makedims -> makedirs"""
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(self.STATIC_DIR, exist_ok=True)  # CORRIGÉ ICI

MODEL_CONFIG = ModelConfig()
PROCESSING_CONFIG = ProcessingConfig()
PATH_CONFIG = PathConfig()