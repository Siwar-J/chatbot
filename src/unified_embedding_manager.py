# src/unified_embedding_manager.py
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class UnifiedEmbeddingManager:
    """Manager unifié pour les embeddings avec support multi-backends"""
    
    def __init__(self, strategy: str = None):
        # UTILISER LA CONFIG EXISTANTE
        from .config import OLLAMA_CONFIG
        
        self.strategy = strategy or OLLAMA_CONFIG.EMBEDDING_STRATEGY
        self.ollama_manager = None
        self.hf_manager = None
        self.is_initialized = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialise le manager selon la stratégie choisie"""
        try:
            logger.info(f"🚀 Initialisation Embedding Manager - Stratégie: {self.strategy}")
            
            
            self._initialize_ollama()
            
            self.is_initialized = True
            logger.info(f"✅ Embedding Manager initialisé: {self.strategy}")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Embedding Manager: {e}")
            self.is_initialized = False
    
    def _initialize_ollama(self) -> bool:
        """Initialise le gestionnaire Ollama"""
        try:
            from .ollama_embedding_manager import OllamaEmbeddingManager
            from .config import OLLAMA_CONFIG
            
            self.ollama_manager = OllamaEmbeddingManager(
                model_name=OLLAMA_CONFIG.OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_CONFIG.OLLAMA_BASE_URL
            )
            
            return self.ollama_manager.is_initialized
            
        except Exception as e:
            logger.warning(f"❌ Ollama Embedding échoué: {e}")
            return False
    
    def encode_documents(self, documents: List[str], **kwargs) -> np.ndarray:
        """Encode les documents avec le backend actif"""
        if not self.is_initialized:
            raise Exception("Embedding Manager non initialisé")
        
        if self.strategy == "ollama" and self.ollama_manager:
            from .config import OLLAMA_CONFIG
            batch_size = kwargs.get('batch_size', OLLAMA_CONFIG.OLLAMA_EMBEDDING_BATCH_SIZE)
            return self.ollama_manager.encode_documents(documents, batch_size)
        else:
            raise Exception("Aucun backend d'embedding disponible")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode une requête"""
        if not self.is_initialized:
            raise Exception("Embedding Manager non initialisé")
        
        if self.strategy == "ollama" and self.ollama_manager:
            return self.ollama_manager.encode_query(query)
        else:
            raise Exception("Aucun backend d'embedding disponible")
    
    def get_embedding_info(self) -> dict:
        """Retourne les informations sur le système d'embedding"""
        info = {
            "strategy": self.strategy,
            "initialized": self.is_initialized
        }
        
        
        info.update(self.ollama_manager.get_model_info())        
        return info
    
    def test_system(self) -> dict:
        """Teste le système d'embedding complet"""
        try:
            if self.strategy == "ollama" and self.ollama_manager:
                return self.ollama_manager.test_embedding()
            else:
                return {"status": "✅ Système initialisé", "strategy": self.strategy}
                
        except Exception as e:
            return {"status": "❌ Erreur test", "error": str(e)}