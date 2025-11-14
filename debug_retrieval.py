# debug_retrieval_final.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_retrieval_system():
    """Script de debug utilisant uniquement config.py"""
    print("🔧 DIAGNOSTIC DU SYSTÈME DE RETRIEVAL (CONFIG UNIQUE)")
    print("=" * 50)
    
    try:
        from src.unified_embedding_manager import UnifiedEmbeddingManager
        from src.advanced_vector_store import AdvancedVectorStore
        from src.config import OLLAMA_CONFIG
        
        print(f"📋 Configuration Ollama:")
        print(f"   - Modèle embedding: {OLLAMA_CONFIG.OLLAMA_EMBEDDING_MODEL}")
        print(f"   - Modèle LLM: {OLLAMA_CONFIG.OLLAMA_LLM_MODEL}")
        print(f"   - Stratégie: {OLLAMA_CONFIG.EMBEDDING_STRATEGY}")
        
        # 1. Test de l'embedding
        print("\n1. TEST EMBEDDING MANAGER")
        print("-" * 30)
        
        embedding_manager = UnifiedEmbeddingManager()
        
        if embedding_manager.is_initialized:
            print("✅ Embedding Manager initialisé")
            embed_info = embedding_manager.get_embedding_info()
            print(f"   Modèle: {embed_info.get('model_name', 'N/A')}")
            print(f"   Stratégie: {embed_info.get('strategy')}")
            
            # Test d'embedding simple
            test_texts = ["test technique", "documentation"]
            try:
                embeddings = embedding_manager.encode_documents(test_texts, batch_size=2)
                print(f"✅ Embeddings générés: {embeddings.shape}")
                if len(embeddings) > 0:
                    print(f"   Sample: {embeddings[0][:3]}...")  # Premières valeurs
            except Exception as e:
                print(f"❌ Erreur génération embeddings: {e}")
                return
        else:
            print("❌ Embedding Manager non initialisé")
            return
        
        # 2. Test du vector store
        print("\n2. TEST VECTOR STORE")
        print("-" * 30)
        
        vector_store = AdvancedVectorStore()
        
        # Vérifier les collections existantes
        try:
            collections = vector_store.client.list_collections()
            print(f"📚 Collections existantes: {[col.name for col in collections]}")
            
            if collections:
                collection_name = collections[0].name
                vector_store.collection = vector_store.client.get_collection(collection_name)
                
                # Compter les documents
                count = vector_store.collection.count()
                print(f"📊 Documents dans la collection: {count}")
                
                if count > 0:
                    # Test de recherche
                    print("\n3. TEST RECHERCHE")
                    print("-" * 30)
                    
                    test_queries = ["installation", "configuration"]
                    
                    for query in test_queries:
                        print(f"\n🔍 Recherche: '{query}'")
                        results = vector_store.similarity_search(query, k=3)
                        print(f"   Résultats trouvés: {len(results)}")
                        
                        for j, result in enumerate(results):
                            content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                            score = result.get('distance', 'N/A')
                            print(f"   - Résultat {j+1}: distance={score:.3f}")
                
                else:
                    print("❌ Aucun document dans la collection!")
                    print("💡 Vous devez d'abord uploader et traiter un document PDF")
                    
            else:
                print("❌ Aucune collection trouvée!")
                print("💡 Vous devez d'abord uploader et traiter un document PDF")
                
        except Exception as e:
            print(f"❌ Erreur vector store: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_retrieval_system()