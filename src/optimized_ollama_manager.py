# src/optimized_ollama_manager.py
import requests
import logging
import time
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger(__name__)

class OptimizedOllamaManager:
    """Gestionnaire Ollama optimisé pour vos modèles disponibles"""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = self._select_best_model(model)
        self.timeout = 180  # 3 minutes pour les documents techniques
        self.is_initialized = False
        
        # Session HTTP optimisée
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Adapter avec retry strategy
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self._initialize()
    
    def _select_best_model(self, preferred_model: str) -> str:
        """Sélectionne le meilleur modèle disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                logger.info(f"📦 Modèles disponibles: {available_models}")
                
                # Priorité des modèles
                model_priority = [
                    preferred_model,
                    "mistral",           # 7B - Rapide et efficace
                    "deepseek-r1",       # 7B - Bon pour le raisonnement
                    "nomic-embed-text"   # Embedding seulement
                ]
                
                for model_name in model_priority:
                    if any(model_name in avail_model for avail_model in available_models):
                        selected = next((m for m in available_models if model_name in m), available_models[0])
                        logger.info(f"✅ Modèle sélectionné: {selected}")
                        return selected
                
                # Fallback au premier modèle disponible
                if available_models:
                    selected = available_models[0]
                    logger.info(f"🔄 Fallback sur: {selected}")
                    return selected
            
            logger.warning("⚠️ Aucun modèle trouvé, utilisation du modèle par défaut")
            return preferred_model
            
        except Exception as e:
            logger.error(f"❌ Erreur sélection modèle: {e}")
            return preferred_model
    
    def _initialize(self):
        """Initialisation rapide"""
        try:
            # Test de connexion rapide
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_initialized = True
                logger.info(f"✅ Ollama initialisé - Modèle: {self.model}")
            else:
                logger.error(f"❌ Erreur HTTP: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
    
    def generate_technical_response(self, context: str, question: str, max_retries: int = 2) -> str:
        """Génération optimisée pour les réponses techniques"""
        try:
            prompt = self._create_technical_prompt(context, question)
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔄 Génération technique - tentative {attempt + 1}")
                    
                    payload = {
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Plus déterministe pour la technique
                            "top_p": 0.9,
                            "top_k": 40,
                            "num_predict": 1024,  # Limiter la longueur
                            "num_thread": 4,
                            "repeat_penalty": 1.1
                        }
                    }
                    
                    response = self.session.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get('response', '').strip()
                        
                        if response_text:
                            logger.info(f"✅ Réponse générée ({len(response_text)} caractères)")
                            return self._post_process_response(response_text)
                        else:
                            logger.warning("⚠️ Réponse vide")
                            return "Je n'ai pas pu générer de réponse pour le moment."
                    
                    else:
                        logger.warning(f"❌ HTTP {response.status_code}: {response.text}")
                        
                except requests.exceptions.Timeout:
                    logger.error(f"⏰ Timeout tentative {attempt + 1}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        logger.info(f"⏳ Attente {wait_time}s...")
                        time.sleep(wait_time)
                    continue
                    
                except requests.exceptions.ConnectionError:
                    logger.error(f"🔌 Connexion perdue tentative {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(10)
                    continue
                    
                except Exception as e:
                    logger.error(f"❌ Erreur tentative {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    continue
            
            return "❌ Service temporairement indisponible. Veuillez réessayer."
            
        except Exception as e:
            logger.error(f"❌ Erreur génération technique: {e}")
            return "Erreur lors de la génération de la réponse."
    
    def _create_technical_prompt(self, context: str, question: str) -> str:
        """Crée un prompt optimisé pour la documentation technique"""
        return f"""Tu es un expert technique assistant des utilisateurs avec de la documentation.

DOCUMENTATION DE RÉFÉRENCE:
{context}

QUESTION DE L'UTILISATEUR:
{question}

INSTRUCTIONS STRICTES:
1. Réponds UNIQUEMENT en français
2. Utilise EXCLUSIVEMENT les informations fournies dans la documentation
3. Si l'information n'est pas dans la documentation, dis clairement "Cette information n'est pas disponible dans la documentation fournie."
4. Sois concis et technique
5. Structure ta réponse avec des points si nécessaire
6. Ne invente AUCUNE information

RÉPONSE:"""
    
    def _post_process_response(self, response: str) -> str:
        """Nettoie et améliore la réponse"""
        # Supprimer les répétitions de prompt
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('DOCUMENTATION') and not line.startswith('QUESTION'):
                cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines)
        
        # Limiter la longueur si nécessaire
        if len(cleaned_response) > 1500:
            cleaned_response = cleaned_response[:1497] + "..."
        
        return cleaned_response
    
    def quick_test(self) -> str:
        """Test rapide de génération"""
        try:
            payload = {
                "model": self.model,
                "prompt": "Réponds 'TEST OK' en français.",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return f"✅ Test réussi: {result.get('response', 'N/A')}"
            else:
                return f"❌ Test échoué: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ Test échoué: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retourne les informations système"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            models = response.json().get('models', []) if response.status_code == 200 else []
            
            return {
                "status": "✅ Connecté" if self.is_initialized else "❌ Déconnecté",
                "model_actuel": self.model,
                "modèles_disponibles": [model['name'] for model in models],
                "timeout_configuré": f"{self.timeout}s",
                "base_url": self.base_url
            }
        except:
            return {"status": "❌ Impossible de récupérer les informations"}