# src/unified_llm_manager.py
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

class UnifiedLLMManager:
    """Manager unifié avec support Ollama, Cloud et Local"""
    
    def __init__(self, mode: str = "auto", ollama_model: str = "mistral"):
        """
        Args:
            mode: "auto", "cloud", "ollama", "local"
            ollama_model: "mistral", "llama2", etc. (si mode=ollama)
        """
        self.mode = mode
        self.ollama_model = ollama_model
        self.cloud_manager = None
        self.ollama_manager = None
        self.local_manager = None
        self.is_initialized = False
        
        logger.info(f"🚀 Initialisation - Mode: {mode}")
        
        self._initialize()
    
    def _initialize(self):
        """Initialisation selon le mode choisi"""
        try:
            if self.mode == "auto":
                self._initialize_auto()
            elif self.mode == "ollama":
                self._initialize_ollama()
            elif self.mode == "cloud":
                self._initialize_cloud()
            elif self.mode == "local":
                self._initialize_local()
            else:
                logger.error(f"❌ Mode non supporté: {self.mode}")
                self._initialize_auto()  # Fallback
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            self._initialize_fallback()
    
    def _initialize_auto(self):
        """Détection automatique: Ollama -> Cloud -> Local"""
        logger.info("🔍 Détection automatique...")
        
        # 1. Essayer Ollama d'abord (meilleure performance)
        if self._initialize_ollama():
            self.mode = "ollama"
            return
        
        # 2. Essayer le cloud
        if self._initialize_cloud():
            self.mode = "cloud"
            return
        
        # 3. Fallback local
        if self._initialize_local():
            self.mode = "local"
            return
        
        logger.error("❌ Aucun mode n'a fonctionné")
    
    def _initialize_ollama(self) -> bool:
        """Initialise Ollama"""
        try:
            from .ollama_manager import OllamaManager
            
            self.ollama_manager = OllamaManager(model=self.ollama_model)
            if self.ollama_manager.is_initialized:
                self.is_initialized = True
                logger.info("✅ Mode Ollama activé")
                return True
            return False
        except Exception as e:
            logger.warning(f"❌ Ollama échoué: {e}")
            return False
    
    def _initialize_local(self) -> bool:
        """Initialise le mode local"""
        try:
            from .ollama_manager import OllamaManager
            from .config import OPTIMIZED_Ollama_CONFIG
            
            self.local_manager = OllamaManager()
            self.local_manager.initialize()
            
            if self.local_manager.is_initialized:
                self.is_initialized = True
                logger.info("✅ Mode local activé")
                return True
            return False
            
        except Exception as e:
            logger.warning(f"❌ Local échoué: {e}")
            return False
    
    def _initialize_fallback(self):
        """Dernier recours"""
        logger.warning("🔄 Activation du fallback local basique...")
        try:
            from .ollama_manager import OllamaManager
            self.local_manager = OllamaManager()
            self.local_manager.initialize()
            self.is_initialized = self.local_manager.is_initialized
        except:
            self.is_initialized = False

    def generate_response(self, prompt: str) -> str:
        if not self.is_initialized:
            return "❌ Système non initialisé"
        
        if self.ollama_manager:
            return self.ollama_manager.generate_response(prompt)
        else:
            return "❌ Aucun manager disponible"
    
    def create_technical_prompt(self, context: str, question: str) -> str:
        if self.ollama_manager:
            return self.ollama_manager.create_prompt(context, question)
        else:
            return f"Contexte: {context}\nQuestion: {question}\nRéponse:"
    
    def get_model_info(self) -> dict:
        if self.ollama_manager:
            info = self.ollama_manager.get_model_info()
            info["mode"] = "ollama"
            return info
        else:
            return {
                "model": "Aucun",
                "device": "Non initialisé",
                "mode": self.mode,
                "initialized": False
            }
    
    def test_generation(self) -> str:
        if self.ollama_manager:
            return self.ollama_manager.test_generation()

        else:
            return "❌ Aucun manager disponible"
    
    def initialize(self, **kwargs):
        pass