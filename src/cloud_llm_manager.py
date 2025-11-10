# src/cloud_llm_manager.py
import os
import requests
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

class CloudLLMManager:
    """Gestionnaire cloud unifié pour tous les providers"""
    
    def __init__(self, provider: str = "huggingface", api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.is_initialized = True  # Toujours initialisé pour cloud
        
        # Configuration des endpoints
        self.endpoints = {
            "huggingface": f"https://api-inference.huggingface.co/models/{self.model or 'mistralai/Mistral-7B-Instruct-v0.2'}",
            "openai": "https://api.openai.com/v1/chat/completions",
            "together": "https://api.together.xyz/v1/completions"
        }
        
        # Configuration des headers
        self.headers = self._setup_headers()
        
        logger.info(f"☁️  Cloud Manager initialisé - Provider: {provider}, Model: {self.model}")
    
    def _setup_headers(self):
        """Configure les headers selon le provider"""
        headers = {"Content-Type": "application/json"}
        
        if self.provider == "huggingface" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "openai" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "together" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """Génère une réponse via le cloud avec retry logic"""
        for attempt in range(max_retries):
            try:
                if self.provider == "huggingface":
                    return self._call_huggingface(prompt, attempt)
                elif self.provider == "openai":
                    return self._call_openai(prompt)
                elif self.provider == "together":
                    return self._call_together(prompt)
                else:
                    return f"❌ Provider non supporté: {self.provider}"
                    
            except Exception as e:
                logger.error(f"❌ Tentative {attempt + 1} échouée: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    logger.info(f"⏳ Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return f"❌ Échec après {max_retries} tentatives: {str(e)}"
        
        return "❌ Échec de génération"
    
    def _call_huggingface(self, prompt: str, attempt: int) -> str:
        """Appel à l'API Hugging Face"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }
        
        response = requests.post(
            self.endpoints["huggingface"],
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 503:
            wait_time = (attempt + 1) * 10
            logger.info(f"⏳ Modèle en chargement, attente {wait_time}s...")
            time.sleep(wait_time)
            raise Exception("Modèle en cours de chargement")
        
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '').strip()
        else:
            return str(result)
    
    def _call_openai(self, prompt: str) -> str:
        """Appel à l'API OpenAI"""
        payload = {
            "model": self.model or "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        response = requests.post(
            self.endpoints["openai"],
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    
    def _call_together(self, prompt: str) -> str:
        """Appel à l'API Together AI"""
        payload = {
            "model": self.model or "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = requests.post(
            self.endpoints["together"],
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['text'].strip()
    
    def create_technical_prompt(self, context: str, question: str) -> str:
        """Prompt optimisé pour les modèles cloud"""
        return f"""En tant qu'assistant technique expert, analyse le contexte suivant et réponds à la question.

CONTEXTE DOCUMENT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Réponds en français de manière structurée et technique
- Utilise UNIQUEMENT les informations du contexte fourni
- Si l'information n'est pas dans le contexte, indique-le clairement
- Sois précis et cite les parties pertinentes
- Évite les répétitions et les informations génériques

RÉPONSE:"""
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle cloud"""
        return {
            "model": self.model or "Cloud Model",
            "device": "☁️  Cloud",
            "initialized": True,
            "provider": self.provider,
            "type": "cloud"
        }
    
    def test_generation(self) -> str:
        """Teste la génération cloud"""
        try:
            test_prompt = "Explique le concept d'apprentissage automatique en 2 phrases."
            response = self.generate_response(test_prompt)
            return f"✅ Cloud ({self.provider}): {response[:100]}..."
        except Exception as e:
            return f"❌ Test cloud échoué: {str(e)}"