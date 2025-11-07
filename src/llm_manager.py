# src/llm_manager.py
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
from langchain_community.llms import HuggingFacePipeline
import gc

logger = logging.getLogger(__name__)

class LLMManager:
    """Gère les modèles légers pour 4GB GPU"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.llm = None
        # Forcer le CPU pour 4GB GPU - plus stable
        self.device = "cpu"  # Même avec GPU, on utilise CPU pour stabilité
        self.is_initialized = False
        
    def initialize(self, **generation_kwargs):
        """
        Initialisation optimisée pour faible mémoire
        """
        try:
            logger.info(f"🚀 Initialisation de {self.model_name} sur {self.device}")
            
            self._clean_memory()
            
            # Paramètres conservateurs pour faible mémoire
            safe_kwargs = {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1,
            }
            safe_kwargs.update(generation_kwargs)
            
            # Chargement du tokenizer
            logger.info("📥 Chargement du tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configuration des tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Chargement du modèle - SUR CPU pour stabilité
            logger.info("📥 Chargement du modèle sur CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=None,  # Pas de device_map pour CPU
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            # Création du pipeline
            logger.info("⚙️ Création du pipeline...")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=safe_kwargs["max_new_tokens"],
                temperature=safe_kwargs["temperature"],
                top_p=safe_kwargs["top_p"],
                repetition_penalty=safe_kwargs["repetition_penalty"],
                do_sample=safe_kwargs["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                device=-1,  # -1 pour CPU
            )
            
            # Intégration avec LangChain
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.is_initialized = True
            
            logger.info(f"✅ {self.model_name} initialisé avec succès sur {self.device}")
            
            # Test rapide
            test_result = self.test_generation()
            logger.info(f"🧪 Test de génération: {test_result}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            self._try_ultra_light_model(**generation_kwargs)
    
    def _try_ultra_light_model(self, **generation_kwargs):
        """Tente un modèle ultra-léger en dernier recours"""
        ultra_light_models = [
            "distilgpt2",           # ~500MB
            "gpt2",                 # ~600MB  
            "microsoft/DialoGPT-small",  # ~400MB
        ]
        
        for light_model in ultra_light_models:
            try:
                logger.info(f"🔄 Tentative avec modèle ultra-léger: {light_model}")
                self.model_name = light_model
                self.llm = None
                self.is_initialized = False
                
                self.initialize(**generation_kwargs)
                if self.is_initialized:
                    logger.info(f"✅ Modèle ultra-léger chargé: {light_model}")
                    return
                    
            except Exception as e:
                logger.warning(f"❌ {light_model} a échoué: {e}")
                continue
        
        logger.error("❌ Aucun modèle n'a pu être chargé")
        self.is_initialized = False
    
    def _clean_memory(self):
        """Nettoie la mémoire"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate_response(self, prompt: str) -> str:
        """Génère une réponse"""
        if not self.is_initialized:
            return "❌ Le modèle n'est pas initialisé. Veuillez réessayer."
        
        try:
            self._clean_memory()
            logger.info(f"🎯 Génération pour: {prompt[:80]}...")
            
            # Utilisation simple et directe
            response = self.llm.invoke(prompt)
            
            cleaned_response = self._clean_response(response)
            logger.info(f"✅ Réponse générée ({len(cleaned_response)} caractères)")
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"❌ Erreur de génération: {e}")
            return f"❌ Erreur lors de la génération: {str(e)}"
    
    def _clean_response(self, response):
        """Nettoie la réponse"""
        if isinstance(response, str):
            text = response
        else:
            text = str(response)
        
        # Nettoyage basique
        text = text.strip()
        
        # Supprimer les éventuelles répétitions du prompt
        lines = text.split('\n')
        clean_lines = [line for line in lines if line.strip()]
        
        return ' '.join(clean_lines)
    
    def create_technical_prompt(self, context: str, question: str) -> str:
        """Crée un prompt simple et efficace"""
        # Format simple pour les petits modèles
        return f"""Document: {context}

Question: {question}

Réponds en français en utilisant uniquement le document. Si tu ne sais pas, dis-le.

Réponse:"""
    
    def test_generation(self) -> str:
        """Teste la génération"""
        if not self.is_initialized:
            return "❌ Modèle non initialisé"
        
        try:
            response = self.generate_response("Bonjour, ça va?")
            if response and len(response) > 10:
                return f"✅ Test réussi: {response[:50]}..."
            else:
                return "❌ Réponse trop courte"
        except Exception as e:
            return f"❌ Test échoué: {e}"
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle"""
        return {
            "model": self.model_name,
            "device": self.device,
            "initialized": self.is_initialized,
            "memory_optimized": "Oui (4GB compatibilité)"
        }