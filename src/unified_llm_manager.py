# src/unified_llm_manager.py
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

class UnifiedLLMManager:
    """Manager unifié avec support Ollama, Cloud et Local"""
    
    def __init__(self, mode: str = "auto", provider: str = "huggingface", ollama_model: str = "mistral"):
        """
        Args:
            mode: "auto", "cloud", "ollama", "local"
            provider: "huggingface", "openai" (si mode=cloud)
            ollama_model: "mistral", "llama2", etc. (si mode=ollama)
        """
        self.mode = mode
        self.provider = provider
        self.ollama_model = ollama_model
        self.cloud_manager = None
        self.ollama_manager = None
        self.local_manager = None
        self.is_initialized = False
        
        logger.info(f"🚀 Initialisation - Mode: {mode}, Provider: {provider}")
        
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
    
    def _initialize_cloud(self) -> bool:
        """Initialise le cloud"""
        try:
            from .cloud_llm_manager import CloudLLMManager
            from .cloud_config import CLOUD_CONFIG
            
            if self.provider == "huggingface":
                api_key = os.getenv("HF_API_KEY")
                model = CLOUD_CONFIG.HF_MODEL
            elif self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                model = CLOUD_CONFIG.OPENAI_MODEL
            else:
                return False
            
            if not api_key:
                return False
            
            self.cloud_manager = CloudLLMManager(
                provider=self.provider,
                api_key=api_key,
                model=model
            )
            
            # Test rapide
            test_result = self.cloud_manager.test_generation()
            if "✅" in test_result:
                self.is_initialized = True
                logger.info("✅ Mode cloud activé")
                return True
            return False
            
        except Exception as e:
            logger.warning(f"❌ Cloud échoué: {e}")
            return False
    
    def _initialize_local(self) -> bool:
        """Initialise le mode local"""
        try:
            from .llm_manager import LLMManager
            from .config import MODEL_CONFIG
            
            self.local_manager = LLMManager(MODEL_CONFIG.LLM_MODEL)
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
            from .llm_manager import LLMManager
            self.local_manager = LLMManager("distilgpt2")
            self.local_manager.initialize()
            self.is_initialized = self.local_manager.is_initialized
        except:
            self.is_initialized = False

    def generate_response(self, prompt: str) -> str:
        if not self.is_initialized:
            return "❌ Système non initialisé"
        
        if self.ollama_manager:
            return self.ollama_manager.generate_response(prompt)
        elif self.cloud_manager:
            return self.cloud_manager.generate_response(prompt)
        elif self.local_manager:
            return self.local_manager.generate_response(prompt)
        else:
            return "❌ Aucun manager disponible"
    
    def create_technical_prompt(self, context: str, question: str) -> str:
        if self.ollama_manager:
            return self.ollama_manager.create_technical_prompt(context, question)
        elif self.cloud_manager:
            return self.cloud_manager.create_technical_prompt(context, question)
        elif self.local_manager:
            return self.local_manager.create_technical_prompt(context, question)
        else:
            return f"Contexte: {context}\nQuestion: {question}\nRéponse:"
    
    def get_model_info(self) -> dict:
        if self.ollama_manager:
            info = self.ollama_manager.get_model_info()
            info["mode"] = "ollama"
            return info
        elif self.cloud_manager:
            info = self.cloud_manager.get_model_info()
            info["mode"] = "cloud"
            return info
        elif self.local_manager:
            info = self.local_manager.get_model_info()
            info["mode"] = "local"
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
        elif self.cloud_manager:
            return self.cloud_manager.test_generation()
        elif self.local_manager:
            return self.local_manager.test_generation()
        else:
            return "❌ Aucun manager disponible"
    
    def initialize(self, **kwargs):
        pass