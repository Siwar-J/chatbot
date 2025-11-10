# src/advanced_vector_store.py
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder
import re

logger = logging.getLogger(__name__)

class AdvancedVectorStore:
    """Vector store avec re-ranking et techniques avancées"""
    
    def __init__(self, embedding_manager, persist_directory: str = "./chroma_db"):
        self.embedding_manager = embedding_manager
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.reranker = None
        self._initialize()
    
    def _initialize(self):
        """Initialise ChromaDB et le re-ranker"""
        try:
            # Client ChromaDB
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialiser le re-ranker (cross-encoder)
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("✅ Re-ranker initialisé")
            except Exception as e:
                logger.warning(f"❌ Re-ranker non disponible: {e}")
                self.reranker = None
            
            logger.info("✅ Vector Store avancé initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Vector Store: {e}")
            raise
    
    def create_collection(self, documents: List[Dict], collection_name: str = "technical_docs"):
        """Crée une collection avec les documents"""
        try:
            # Supprimer la collection existante si elle existe
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            
            # Créer une nouvelle collection
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Documentation technique"}
            )
            
            # Préparer les données
            texts = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Générer les embeddings
            embeddings = self.embedding_manager.encode_documents(texts)
            
            # Ajouter à la collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Collection créée avec {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"❌ Erreur création collection: {e}")
            raise
    
    def advanced_similarity_search(
        self, 
        query: str, 
        k: int = 10,
        rerank_top_k: int = 5,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Recherche avancée avec re-ranking et filtrage
        """
        try:
            logger.info(f"🔍 Recherche avancée: '{query}'")
            
            # Étape 1: Recherche vectorielle de base
            base_results = self._base_vector_search(query, k * 2, filters)
            
            if not base_results:
                return []
            
            # Étape 2: Re-ranking avec cross-encoder
            if self.reranker and len(base_results) > 1:
                reranked_results = self._rerank_results(query, base_results, rerank_top_k)
            else:
                reranked_results = base_results[:rerank_top_k]
            
            # Étape 3: Filtrage par pertinence
            filtered_results = self._filter_by_relevance(reranked_results, query)
            
            logger.info(f"✅ {len(filtered_results)} résultats pertinents trouvés")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche avancée: {e}")
            return self._fallback_search(query, k)
    
    def _base_vector_search(self, query: str, k: int, filters: Dict = None) -> List[Dict]:
        """Recherche vectorielle de base"""
        try:
            query_embedding = self.embedding_manager.encode_query(query)
            
            search_params = {
                "query_embeddings": query_embedding.tolist(),
                "n_results": k
            }
            
            if filters:
                search_params["where"] = filters
            
            results = self.collection.query(**search_params)
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'id': results['ids'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche vectorielle: {e}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """Re-rank les résultats avec un cross-encoder"""
        try:
            # Préparer les paires query-document
            pairs = [[query, result['content']] for result in results]
            
            # Calculer les scores de pertinence
            scores = self.reranker.predict(pairs)
            
            # Associer les scores aux résultats
            for i, score in enumerate(scores):
                results[i]['relevance_score'] = float(score)
            
            # Trier par score de pertinence
            reranked = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.warning(f"❌ Re-ranking échoué: {e}")
            return results[:top_k]
    
    def _filter_by_relevance(self, results: List[Dict], query: str, threshold: float = 0.3) -> List[Dict]:
        """Filtre les résultats par pertinence"""
        filtered = []
        
        for result in results:
            score = result.get('relevance_score', result.get('distance', 0))
            
            # Si on a un score de re-ranking
            if 'relevance_score' in result:
                if score > threshold:
                    filtered.append(result)
            # Si on utilise la distance vectorielle
            else:
                if score < 0.5:  # Distance cosinus < 0.5
                    filtered.append(result)
        
        return filtered
    
    def _fallback_search(self, query: str, k: int) -> List[Dict]:
        """Recherche de fallback basée sur le texte"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'id': results['ids'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Fallback search échoué: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Recherche hybride : vectorielle + keyword matching
        """
        try:
            # Recherche vectorielle
            vector_results = self.advanced_similarity_search(query, k)
            
            # Recherche par mots-clés (fallback)
            keyword_results = self._keyword_search(query, k)
            
            # Fusionner et dédupliquer
            all_results = vector_results + keyword_results
            seen_ids = set()
            merged_results = []
            
            for result in all_results:
                doc_id = result.get('id')
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged_results.append(result)
                
                if len(merged_results) >= k:
                    break
            
            return merged_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche hybride: {e}")
            return self.advanced_similarity_search(query, k)
    
    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """Recherche simple par mots-clés"""
        try:
            # Extraire les mots-clés importants
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            # Recherche dans tous les documents
            all_docs = self.collection.get()
            results = []
            
            for i, doc in enumerate(all_docs['documents']):
                score = self._calculate_keyword_score(doc, keywords)
                if score > 0:
                    results.append({
                        'content': doc,
                        'metadata': all_docs['metadatas'][i],
                        'id': all_docs['ids'][i],
                        'keyword_score': score
                    })
            
            # Trier par score
            results.sort(key=lambda x: x['keyword_score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.warning(f"❌ Keyword search échoué: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extrait les mots-clés importants de la requête"""
        # Supprimer les mots vides et garder les termes techniques
        stop_words = {'quel', 'quelle', 'quels', 'quelles', 'comment', 'pourquoi', 'est', 'sont'}
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', query.lower())
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
    
    def _calculate_keyword_score(self, document: str, keywords: List[str]) -> float:
        """Calcule un score basé sur la présence de mots-clés"""
        doc_lower = document.lower()
        score = 0
        
        for keyword in keywords:
            if keyword in doc_lower:
                # Score plus élevé si le mot-clé apparaît plusieurs fois
                count = doc_lower.count(keyword)
                score += min(count * 0.1, 0.5)  # Maximum 0.5 par mot-clé
        
        return score