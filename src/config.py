# src/config.py
import os
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL: str = "distilgpt2"

@dataclass
class ProcessingConfig:
    """Configuration du traitement par sections"""
    # Stratégie de chunking
    CHUNKING_STRATEGY: Literal["sections", "semantic", "hybrid"] = "sections"
    
    # Paramètres pour le chunking par sections
    MIN_SECTION_LENGTH: int = 200    # Caractères minimum par section
    MAX_SECTION_LENGTH: int = 1500   # Caractères maximum par section
    
    # Recherche
    SIMILARITY_TOP_K: int = 4
    
    # Filtrage des sections
    PRIORITY_SECTIONS: list = None
    
    def __post_init__(self):
        if self.PRIORITY_SECTIONS is None:
            self.PRIORITY_SECTIONS = ['installation', 'api', 'troubleshooting', 'examples', 'configuration']

@dataclass
class PathConfig:
    UPLOAD_DIR: str = "data/uploaded_docs"
    VECTOR_STORE_DIR: str = "data/vector_stores"
    STATIC_DIR: str = "static"
    
    def create_directories(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(self.STATIC_DIR, exist_ok=True)

@dataclass
class RetrievalConfig:
    """Configuration avancée pour le retrieval"""
    
    # Embedding
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_STRATEGY: Literal["multilingual", "french", "english"] = "multilingual"
    
    # Retrieval
    SEARCH_STRATEGY: Literal["advanced", "hybrid", "vector"] = "advanced"
    INITIAL_RESULTS: int = 20
    FINAL_RESULTS: int = 5
    RERANK_ENABLED: bool = True
    
    # Filtrage
    MIN_RELEVANCE_SCORE: float = 0.3
    ENABLE_METADATA_FILTERING: bool = True
    
    # Performance
    BATCH_SIZE: int = 32
    ENABLE_CACHING: bool = True

@dataclass
class ChunkingConfig:
    """Configuration du chunking technique"""
    STRATEGY: Literal["sections", "semantic", "technical"] = "technical"
    MIN_SECTION_LENGTH: int = 150
    MAX_SECTION_LENGTH: int = 1200
    PRIORITIZE_TECHNICAL_SECTIONS: bool = True

RETRIEVAL_CONFIG = RetrievalConfig()
CHUNKING_CONFIG = ChunkingConfig()
MODEL_CONFIG = ModelConfig()
PROCESSING_CONFIG = ProcessingConfig()
PATH_CONFIG = PathConfig()