# src/ollama_embedding_manager.py
import logging
import requests
import numpy as np
from typing import List, Optional
import time

from .config import OLLAMA_CONFIG


logger = logging.getLogger(__name__)

class OllamaEmbeddingManager:
    """Gestionnaire d'embeddings utilisant Ollama"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        
        # MODIFICATION: Toujours utiliser nomic-embed-text
        self.model_name = model_name or OLLAMA_CONFIG.OLLAMA_EMBEDDING_MODEL  # nomic-embed-text
        self.base_url = base_url or OLLAMA_CONFIG.OLLAMA_BASE_URL
        self.timeout = OLLAMA_CONFIG.OLLAMA_TIMEOUT
        
        # MODIFICATION: Dimensions FIXES depuis la config
        self.dimensions = OLLAMA_CONFIG.OLLAMA_EMBEDDING_DIMENSIONS  # 768
        
        self.is_initialized = False
        self._initialize()
    
    def _initialize(self):
        """Initialise avec validation des dimensions"""
        try:
            logger.info(f"🚀 Initialisation Ollama Embedding: {self.model_name} ({self.dimensions}D)")
            
            # Vérifier que Ollama est accessible
            self._check_ollama_connection()
            
            # Vérifier que le modèle d'embedding est disponible
            self._check_embedding_model()
            
            # MODIFICATION: Test de validation des dimensions
            test_embedding = self._generate_embeddings(["test validation dimensions"])
            if test_embedding is not None and len(test_embedding) > 0:
                actual_dims = len(test_embedding[0])
                if actual_dims != self.dimensions:
                    logger.error(f"❌ Dimensions modèle incorrectes: {actual_dims}D ≠ {self.dimensions}D")
                    raise ValueError(f"Dimensions modèle incorrectes: {actual_dims} au lieu de {self.dimensions}")
                
                self.is_initialized = True
                logger.info(f"✅ Ollama Embedding validé: {self.model_name} ({actual_dims}D)")
            else:
                raise Exception("Échec du test d'embedding")
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Ollama Embedding: {e}")
            self.is_initialized = False
    
    def _check_ollama_connection(self):
        """Vérifie la connexion à Ollama"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            if response.status_code != 200:
                raise Exception(f"Ollama non accessible: {response.status_code}")
            
            logger.info("✅ Connexion Ollama établie")
            
        except requests.exceptions.ConnectionError:
            raise Exception("Ollama n'est pas démarré. Lancez: ollama serve")
        except Exception as e:
            raise Exception(f"Erreur connexion Ollama: {e}")
    
    def _check_embedding_model(self):
        """Vérifie que le modèle d'embedding est disponible"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            # Vérifier si le modèle est disponible
            model_found = any(
                self.model_name in model_name or model_name == self.model_name
                for model_name in available_models
            )
            
            if not model_found:
                logger.warning(f"⚠️  Modèle {self.model_name} non trouvé. Tentative de téléchargement...")
                self._pull_embedding_model()
            else:
                logger.info(f"✅ Modèle d'embedding trouvé: {self.model_name}")
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification modèle: {e}")
            raise
    
    def _pull_embedding_model(self):
        """Télécharge le modèle d'embedding s'il n'est pas disponible"""
        try:
            logger.info(f"📥 Téléchargement du modèle: {self.model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300  # 5 minutes pour le téléchargement
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Modèle {self.model_name} téléchargé avec succès")
            else:
                raise Exception(f"Échec téléchargement: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement modèle: {e}")
            
            # Suggérer des alternatives
            available_models = self.get_available_models()
            if available_models:
                alternative = available_models[0]
                logger.info(f"🔄 Utilisation alternative: {alternative}")
                self.model_name = alternative
            else:
                raise Exception("Aucun modèle d'embedding disponible")
    
    def _generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de textes"""
        try:
            embeddings = []
            
            for text in texts:
                if not text.strip():
                    # Embedding nul pour les textes vides
                    embeddings.append(np.zeros(self.dimensions))
                    continue
                
                payload = {
                    "model": self.model_name,
                    "prompt": text
                }
                
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding', [])
                    if len(embedding) != self.dimensions:
                        logger.warning(f"⚠️  Dimension d'embedding inattendue: {len(embedding)} au lieu de {self.dimensions}")
                    embeddings.append(np.array(embedding))
                else:
                    logger.error(f"❌ Erreur embedding: {response.status_code} - {response.text}")
                    return None
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"❌ Erreur génération embeddings: {e}")
            return None
    
    def encode_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode une liste de documents avec gestion des batches"""
        try:
            if not self.is_initialized:
                raise Exception("Ollama Embedding non initialisé")
            
            logger.info(f"🔤 Encodage de {len(documents)} documents avec {self.model_name}...")
            
            all_embeddings = []
            
            # Traitement par batch pour éviter les timeouts
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.debug(f"Traitement batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                batch_embeddings = self._generate_embeddings(batch)
                if batch_embeddings is not None:
                    all_embeddings.append(batch_embeddings)
                else:
                    # Créer des embeddings nuls pour ce batch
                    null_embeddings = np.zeros((len(batch), self.dimensions))
                    all_embeddings.append(null_embeddings)
                
                # Pause courte entre les batches
                if i + batch_size < len(documents):
                    time.sleep(0.1)
            
            if not all_embeddings:
                raise Exception("Aucun embedding généré")
            
            embeddings = np.vstack(all_embeddings)
            logger.info(f"✅ Embeddings générés: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Erreur encodage documents: {e}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode une requête unique"""
        try:
            if not self.is_initialized:
                raise Exception("Ollama Embedding non initialisé")
            
            embeddings = self._generate_embeddings([query])
            if embeddings is not None and len(embeddings) > 0:
                return embeddings[0]
            else:
                raise Exception("Échec génération embedding requête")
                
        except Exception as e:
            logger.error(f"❌ Erreur encodage requête: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Retourne la liste des modèles d'embedding disponibles"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            models_data = response.json()
            all_models = [model['name'] for model in models_data.get('models', [])]
            
            # Filtrer les modèles d'embedding (basé sur la config)
            from .config import OLLAMA_CONFIG
            embedding_models = []
            for model_name in all_models:
                for embedding_model in OLLAMA_CONFIG.OLLAMA_EMBEDDING_MODELS:
                    if embedding_model in model_name:
                        embedding_models.append(model_name)
            
            return embedding_models
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "base_url": self.base_url,
            "initialized": self.is_initialized,
            "available_models": self.get_available_models()
        }
    
    def test_embedding(self) -> dict:
        """Teste le système d'embedding"""
        try:
            test_texts = [
                "Test d'embedding en français",
                "Embedding test in English"
            ]
            
            embeddings = self.encode_documents(test_texts, batch_size=2)
            
            return {
                "status": "✅ Succès",
                "model": self.model_name,
                "dimensions": embeddings.shape[1],
                "samples": len(test_texts),
                "embedding_sample": embeddings[0][:5].tolist()  # Premières 5 dimensions
            }
            
        except Exception as e:
            return {
                "status": "❌ Échec",
                "error": str(e)
            }