# src/ollama_config.py
from dataclasses import dataclass

@dataclass
class OllamaConfig:
    """Configuration Ollama"""
    DEFAULT_MODEL: str = "mistral"
    BASE_URL: str = "http://localhost:11434"
    TIMEOUT: int = 60
    
    # Modèles recommandés
    RECOMMENDED_MODELS = [
        "mistral",      # 7B, excellent équilibre
        "llama2",       # 7B, très bon
        "codellama",    # 7B/13B, bon pour le code
        "mixtral",      # 8x7B, très performant (si assez de RAM)
        "phi",          # 2.7B, très léger
    ]

OLLAMA_CONFIG = OllamaConfig()