# src/ollama_manager.py
import requests
import logging
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OllamaManager:
    """Gère les modèles via Ollama"""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.is_initialized = False
        self._initialize()
    
    def _initialize(self):
        """Vérifie qu'Ollama fonctionne"""
        try:
            # Test de connexion
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model in model_names or any(self.model in name for name in model_names):
                    self.is_initialized = True
                    logger.info(f"✅ Ollama initialisé - Modèle: {self.model}")
                else:
                    logger.error(f"❌ Modèle {self.model} non trouvé. Modèles disponibles: {model_names}")
            else:
                logger.error(f"❌ Ollama non accessible: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ Ollama non démarré. Lancez: ollama serve")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Ollama: {e}")
    
    def generate_response(self, prompt: str) -> str:
        """Génère une réponse via Ollama"""
        if not self.is_initialized:
            return "❌ Ollama non initialisé"
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1024
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"❌ Erreur génération Ollama: {e}")
            return f"❌ Erreur: {str(e)}"
    
    def chat_completion(self, messages: list) -> str:
        """Version chat (plus naturelle)"""
        if not self.is_initialized:
            return "❌ Ollama non initialisé"
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"❌ Erreur chat Ollama: {e}")
            return self.generate_response(messages[-1]['content'])  # Fallback
    
    def create_technical_prompt(self, context: str, question: str) -> str:
        """Prompt optimisé pour Ollama"""
        return f"""En tant qu'assistant technique expert, analyse le contexte documentaire suivant et réponds à la question.

CONTEXTE DOCUMENTAIRE:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Réponds en français de manière structurée et technique
- Utilise UNIQUEMENT les informations du contexte fourni
- Si l'information n'est pas dans le contexte, indique-le clairement
- Sois précis et cite les parties pertinentes du document
- Évite les répétitions et les informations génériques

RÉPONSE TECHNIQUE:"""
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle"""
        return {
            "model": self.model,
            "device": "💻 Ollama",
            "initialized": self.is_initialized,
            "type": "ollama"
        }
    
    def test_generation(self) -> str:
        """Teste la génération"""
        if not self.is_initialized:
            return "❌ Ollama non initialisé"
        
        try:
            response = self.generate_response("Explique l'IA en une phrase.")
            return f"✅ Ollama ({self.model}): {response[:80]}..."
        except Exception as e:
            return f"❌ Test échoué: {e}"