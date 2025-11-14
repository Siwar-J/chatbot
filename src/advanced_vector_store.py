# src/advanced_vector_store.py
import logging
import numpy as np
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from .unified_embedding_manager import UnifiedEmbeddingManager
from .config import OLLAMA_CONFIG

logger = logging.getLogger(__name__)

class AdvancedVectorStore:
    """Vector store avec support Ollama"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.embedding_manager = UnifiedEmbeddingManager(
            strategy=OLLAMA_CONFIG.EMBEDDING_STRATEGY
        )
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        
        # MODIFICATION: Dimensions FIXES depuis la config
        self.expected_dimensions = OLLAMA_CONFIG.OLLAMA_EMBEDDING_DIMENSIONS  # 768
        
        self._initialize()
    
    def _initialize(self):
        """Initialise ChromaDB"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            logger.info(f"✅ Vector Store initialisé - Dimensions: {self.expected_dimensions}")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Vector Store: {e}")
            raise
    
    def create_collection(self, documents: List[Dict], collection_name: str = "technical_docs"):
        """Crée une collection avec dimensions garanties"""
        try:
            # MODIFICATION: Toujours supprimer la collection existante
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"🗑️  Ancienne collection '{collection_name}' supprimée")
            except:
                logger.info(f"📝 Création nouvelle collection '{collection_name}'")
            
            # MODIFICATION: Ajouter les métadonnées de dimensions
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Documentation technique",
                    "embedding_model": OLLAMA_CONFIG.OLLAMA_EMBEDDING_MODEL,  # nomic-embed-text
                    "embedding_dimensions": self.expected_dimensions,  # 768
                    "created_with": "nomic-embed-text-768d"
                }
            )
            
            # Préparer les données
            texts = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # MODIFICATION: Log des dimensions attendues
            logger.info(f"🔤 Génération embeddings avec {self.expected_dimensions} dimensions...")
            embeddings = self.embedding_manager.encode_documents(texts)
            
            # MODIFICATION: Validation critique des dimensions
            actual_dimensions = embeddings.shape[1]
            if actual_dimensions != self.expected_dimensions:
                logger.error(f"❌ CRITIQUE: Dimensions incohérentes! {actual_dimensions} ≠ {self.expected_dimensions}")
                raise ValueError(f"Dimensions embedding incohérentes: {actual_dimensions} au lieu de {self.expected_dimensions}")
            
            # Ajouter à la collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Collection '{collection_name}' créée: {len(documents)} docs, {actual_dimensions}D")
            
        except Exception as e:
            logger.error(f"❌ Erreur création collection: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Recherche avec dimensions cohérentes"""
        try:
            logger.info(f"🔍 Recherche: '{query}' (dimensions: {self.expected_dimensions})")
            
            if not self.collection:
                raise Exception("Aucune collection n'est chargée")
            
            # MODIFICATION: Vérification renforcée des dimensions
            collection_dims = self.collection.metadata.get("embedding_dimensions")
            if collection_dims and collection_dims != self.expected_dimensions:
                logger.error(f"❌ INCOMPATIBILITÉ: collection={collection_dims}D, attendu={self.expected_dimensions}D")
                return []
            
            # Encoder la requête (utilisera le MÊME modèle nomic-embed-text)
            query_embedding = self.embedding_manager.encode_query(query)
            
            # MODIFICATION: Validation dimensions requête
            if len(query_embedding) != self.expected_dimensions:
                logger.error(f"❌ Dimensions requête incorrectes: {len(query_embedding)}D ≠ {self.expected_dimensions}D")
                return []
            
            # Rechercher
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'id': results['ids'][0][i]
                    })
            
            logger.info(f"✅ {len(formatted_results)} résultats trouvés")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return []
    
    def advanced_similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Recherche avancée (alias pour compatibilité)"""
        return self.similarity_search(query, k)
    
    def get_collection_info(self) -> Dict:
        """Retourne des informations sur la collection"""
        try:
            if not self.collection:
                return {"error": "Aucune collection chargée"}
            
            count = self.collection.count()
            return {
                "collection_name": self.collection.name,
                "document_count": count,
                "embedding_strategy": self.embedding_manager.get_embedding_info().get('strategy', 'unknown')
            }
        except Exception as e:
            return {"error": str(e)}