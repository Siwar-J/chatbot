# src/cloud_config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class CloudConfig:
    """Configuration cloud optimisée pour Mistral"""
    
    # Forcer Hugging Face avec Mistral
    CLOUD_PROVIDER: str = "huggingface"
    
    # Mistral via Hugging Face
    HF_API_KEY: Optional[str] = os.getenv("HF_API_KEY")
    HF_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"  # Forcé à Mistral
    
    # Désactiver les autres providers si non utilisés
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    TOGETHER_API_KEY: Optional[str] = None
    TOGETHER_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    # Paramètres optimisés pour Mistral
    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TIMEOUT: int = 30

CLOUD_CONFIG = CloudConfig()