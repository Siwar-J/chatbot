# src/vector_store.py
import os
import logging
from typing import List, Optional
import torch

# Gestion des imports compatibles
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Gère la base de données vectorielle"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialise le modèle d'embedding"""
        try:
            logger.info(f"Initialisation des embeddings sur {self.device}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings initialisés avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des embeddings: {str(e)}")
            raise
    
    def create_vector_store(self, documents: List[Document], persist_directory: str) -> Chroma:
        """
        Crée une nouvelle base vectorielle
        
        Args:
            documents: Liste de documents à vectoriser
            persist_directory: Dossier de sauvegarde
            
        Returns:
            Instance Chroma
        """
        try:
            logger.info(f"Création de la base vectorielle avec {len(documents)} documents")
            
            # Vérification que nous avons des documents
            if not documents:
                raise ValueError("Aucun document à ajouter à la base vectorielle")
            
            # Création de la base
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            
            # Vérification immédiate que les documents sont bien ajoutés
            count = self._get_document_count()
            logger.info(f"Base vectorielle créée dans {persist_directory} avec {count} documents")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du vector store: {str(e)}")
            raise
    
    def load_vector_store(self, persist_directory: str) -> Chroma:
        """
        Charge une base vectorielle existante
        
        Args:
            persist_directory: Dossier de la base vectorielle
            
        Returns:
            Instance Chroma
        """
        try:
            if not os.path.exists(persist_directory):
                raise FileNotFoundError(f"Base vectorielle non trouvée: {persist_directory}")
            
            logger.info(f"Chargement de la base vectorielle depuis {persist_directory}")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            
            # Vérification du chargement
            count = self._get_document_count()
            logger.info(f"Base vectorielle chargée depuis {persist_directory} avec {count} documents")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du vector store: {str(e)}")
            raise
    
    def _get_document_count(self) -> int:
        """
        Compte le nombre de documents dans la base vectorielle
        Méthode robuste qui essaie plusieurs approches
        """
        if self.vector_store is None:
            return 0
        
        try:
            # Méthode 1: Utiliser la méthode get() de Chroma
            try:
                all_data = self.vector_store.get()
                if all_data and 'documents' in all_data:
                    count = len(all_data['documents'])
                    logger.info(f"Méthode get() - {count} documents trouvés")
                    return count
            except Exception as e:
                logger.warning(f"Méthode get() a échoué: {e}")
            
            # Méthode 2: Utiliser l'attribut _collection
            try:
                if hasattr(self.vector_store, '_collection') and self.vector_store._collection:
                    count = self.vector_store._collection.count()
                    logger.info(f"Méthode _collection.count() - {count} documents trouvés")
                    return count
            except Exception as e:
                logger.warning(f"Méthode _collection.count() a échoué: {e}")
            
            # Méthode 3: Faire une recherche test
            try:
                test_results = self.vector_store.similarity_search("test", k=1000)
                count = len(test_results)
                logger.info(f"Méthode similarity_search test - {count} documents trouvés")
                return count
            except Exception as e:
                logger.warning(f"Méthode similarity_search test a échoué: {e}")
            
            # Si toutes les méthodes échouent
            logger.warning("Impossible de compter les documents avec les méthodes standards")
            return 0
            
        except Exception as e:
            logger.error(f"Erreur lors du comptage des documents: {e}")
            return 0
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Recherche des documents similaires
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats
            
        Returns:
            Liste de documents similaires
        """
        if self.vector_store is None:
            error_msg = "Base vectorielle non initialisée. Veuillez d'abord charger ou créer une base."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info(f"Recherche de {k} documents similaires pour: {query[:50]}...")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Recherche terminée, {len(results)} résultats trouvés")
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {str(e)}")
            raise
    
    def get_collection_info(self) -> dict:
        """Retourne des informations sur la collection"""
        if self.vector_store is None:
            return {"count": 0, "exists": False}
        
        try:
            count = self._get_document_count()
            return {
                "count": count,
                "exists": True,
                "method": "robust_count"
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos: {str(e)}")
            return {"count": 0, "exists": False}
    
    def test_search(self) -> List[Document]:
        """
        Teste la base avec une recherche simple
        """
        if self.vector_store is None:
            return []
        
        try:
            return self.similarity_search("test", k=2)
        except Exception as e:
            logger.error(f"Test search failed: {e}")
            return []