# src/retrieval_manager.py
import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class RetrievalManager:
    """Manager principal pour l'embedding et le retrieval"""
    
    def __init__(self):
        self.embedding_manager = None
        self.vector_store = None
        self._initialize()
    
    def _initialize(self):
        """Initialise tous les composants"""
        try:
            from .ollama_embedding_manager import OllamaEmbeddingManager
            from .advanced_vector_store import AdvancedVectorStore
            from .config import OLLAMA_CONFIG
            
            # Embedding Manager
            self.embedding_manager = OllamaEmbeddingManager(
                model_name=OLLAMA_CONFIG.OLLAMA_EMBEDDING_MODEL
            )
            
            # Vector Store
            self.vector_store = AdvancedVectorStore(
                        )
            
            logger.info("✅ Retrieval Manager initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Retrieval Manager: {e}")
            raise
    
    def index_documents(self, documents: List[Dict], collection_name: str = "technical_docs"):
        """Indexe les documents dans la base vectorielle"""
        try:
            # Préparer les documents pour l'indexation
            processed_docs = []
            
            for i, doc in enumerate(documents):
                processed_docs.append({
                    'content': doc.page_content,
                    'metadata': {
                        **doc.metadata,
                        'doc_id': f"doc_{i}",
                        'section_type': doc.metadata.get('section_type', 'content'),
                        'content_length': len(doc.page_content)
                    }
                })
            
            # Créer la collection
            self.vector_store.create_collection(processed_docs, collection_name)
            
            logger.info(f"✅ {len(documents)} documents indexés")
            
        except Exception as e:
            logger.error(f"❌ Erreur indexation: {e}")
            raise
    
    def search(self, query: str, k: int = 5, filters: Dict = None) -> List[Dict]:
        """Recherche avancée avec tous les optimisations"""
        try:
            results = self.vector_store.advanced_similarity_search(
                    query, 
                    k
                )
            
            # Post-processing des résultats
            processed_results = self._post_process_results(results, query)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return []
    
    def _post_process_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Post-processing des résultats pour améliorer la qualité"""
        processed = []
        
        for result in results:
            # Filtrer les résultats trop courts
            if len(result['content']) < 50:
                continue
            
            # Ajouter des métadonnées de scoring
            result['final_score'] = self._calculate_final_score(result, query)
            
            # Formater le contenu
            result['formatted_content'] = self._format_content(result['content'])
            
            processed.append(result)
        
        # Trier par score final
        processed.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return processed
    
    def _calculate_final_score(self, result: Dict, query: str) -> float:
        """Calcule un score final combiné"""
        base_score = 0
        
        # Score de similarité vectorielle
        if 'relevance_score' in result:
            base_score += result['relevance_score'] * 0.7
        elif 'distance' in result:
            base_score += (1 - result['distance']) * 0.7
        
        # Bonus pour les sections techniques importantes
        section_type = result['metadata'].get('section_type', 'content')
        if section_type in ['api', 'installation', 'troubleshooting']:
            base_score += 0.2
        
        # Bonus pour la longueur appropriée
        content_length = len(result['content'])
        if 200 <= content_length <= 1000:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _format_content(self, content: str) -> str:
        """Formate le contenu pour l'affichage"""
        # Limiter la longueur
        if len(content) > 500:
            content = content[:497] + "..."
        
        return content