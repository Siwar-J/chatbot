# src/optimized_config.py
from dataclasses import dataclass

@dataclass
class OptimizedConfig:
    """Configuration optimisée pour vos modèles disponibles"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_PREFERRED_MODEL: str = "mistral"  # Votre meilleur modèle
    
    # Embedding Configuration  
    EMBEDDING_MODEL: str = "nomic-embed-text"  # Vous l'avez déjà !
    
    # Processing
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RETRIES: int = 2
    
    # Performance
    BATCH_SIZE: int = 4
    ENABLE_STREAMING: bool = False  # Désactivé pour plus de stabilité

OPTIMIZED_CONFIG = OptimizedConfig()