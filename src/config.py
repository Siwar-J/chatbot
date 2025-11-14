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

# NOUVELLE CONFIGURATION OLLAMA
@dataclass
class OllamaConfig:
    """Configuration pour les modèles Ollama"""
    
    # Activation Ollama
    USE_OLLAMA: bool = True
    
    # Configuration de base Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120
    
    # Modèles Ollama pour l'embedding - FORCER nomic-embed-text
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_DIMENSIONS: int = 768  # Dimensions fixes pour nomic
    
    # Modèles Ollama pour le LLM
    OLLAMA_LLM_MODEL: str = "mistral"
    
    # Stratégie d'embedding - FORCER ollama
    EMBEDDING_STRATEGY: Literal["ollama"] = "ollama"
    
    # Performance
    OLLAMA_EMBEDDING_BATCH_SIZE: int = 8
    OLLAMA_LLM_TIMEOUT: int = 120
    
    # Fallback models
    FALLBACK_EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    FALLBACK_LLM_MODEL: str = "distilgpt2"
    
    def __post_init__(self):
        # Liste des modèles d'embedding Ollama recommandés
        self.OLLAMA_EMBEDDING_MODELS = [
            "nomic-embed-text",      # 768 dimensions
            "all-minilm",            # 384 dimensions  
            "bge-m3",                # 1024 dimensions
        ]
        
        # Liste des modèles LLM Ollama recommandés
        self.OLLAMA_LLM_MODELS = [
            "mistral",              # Excellent équilibre
            "llama2",               # Très bon pour le dialogue
            "codellama",            # Spécialisé code
            "mixtral",              # Très performant
            "phi",                  # Léger et rapide
        ]
@dataclass
class OptimizedOllamaConfig:
    """Configuration optimisée pour vos modèles disponibles"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_PREFERRED_MODEL: str = "mistral"  # Votre meilleur modèle
    
    # Embedding Configuration  
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"  # Vous l'avez déjà !
    
    # Processing
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RETRIES: int = 2
    
    # Performance
    BATCH_SIZE: int = 4
    ENABLE_STREAMING: bool = False  # Désactivé pour plus de stabilité

OPTIMIZED_Ollama_CONFIG = OptimizedOllamaConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
CHUNKING_CONFIG = ChunkingConfig()
MODEL_CONFIG = ModelConfig()
PROCESSING_CONFIG = ProcessingConfig()
PATH_CONFIG = PathConfig()
OLLAMA_CONFIG = OllamaConfig()  # NOUVELLE INSTANCE