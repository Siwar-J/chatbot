# src/embedding_manager.py
import logging
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedEmbeddingManager:
    """Gestionnaire avancé pour les embeddings"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialise le modèle d'embedding avec optimisation"""
        try:
            logger.info(f"🚀 Chargement du modèle d'embedding: {self.model_name}")
            
            # Modèles recommandés par ordre de performance :
            recommended_models = [
                "intfloat/multilingual-e5-large",      # Excellent pour le multilingue
                "sentence-transformers/all-mpnet-base-v2",  # Très bon pour l'anglais
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "dangvantuan/sentence-camembert-large",  # Spécialisé français
            ]
            
            # Essayer les modèles dans l'ordre jusqu'à ce qu'un fonctionne
            for model in [self.model_name] + recommended_models:
                try:
                    self.model = SentenceTransformer(
                        model,
                        device=self.device,
                        trust_remote_code=True
                    )
                    
                    # Test du modèle
                    test_embedding = self.model.encode(["test"])
                    if test_embedding is not None and len(test_embedding) > 0:
                        logger.info(f"✅ Modèle chargé: {model} sur {self.device}")
                        self.model_name = model
                        break
                        
                except Exception as e:
                    logger.warning(f"❌ Échec chargement {model}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("Aucun modèle d'embedding n'a pu être chargé")
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation embedding: {e}")
            raise
    
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode les documents avec optimisation"""
        try:
            logger.info(f"🔤 Encodage de {len(documents)} documents...")
            
            # Préparation des textes
            processed_texts = [self._preprocess_text(doc) for doc in documents]
            
            # Encodage par batch pour optimiser la mémoire
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True  # Important pour la similarité cosinus
            )
            
            logger.info(f"✅ Embeddings générés: {embeddings.shape}")
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"❌ Erreur encodage: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Prétraitement avancé du texte"""
        if not text:
            return ""
        
        # Nettoyage de base
        text = ' '.join(text.split())
        
        # Ajout d'instructions pour les modèles E5
        if "e5" in self.model_name.lower():
            text = f"passage: {text}"
        
        return text
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode une requête avec formatage spécifique"""
        try:
            # Formatage spécial pour les requêtes avec les modèles E5
            if "e5" in self.model_name.lower():
                query = f"query: {query}"
            
            embedding = self.model.encode(
                [query],
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            return embedding.cpu().numpy()
            
        except Exception as e:
            logger.error(f"❌ Erreur encodage requête: {e}")
            raise

# Configuration des embeddings
EMBEDDING_CONFIGS = {
    "multilingual": "intfloat/multilingual-e5-large",
    "french": "dangvantuan/sentence-camembert-large", 
    "english": "sentence-transformers/all-mpnet-base-v2",
    "balanced": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
}