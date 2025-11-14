# app.py
import streamlit as st
import os
import logging
from datetime import datetime
import tempfile

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import des nouveaux composants
from src.retrieval_manager import RetrievalManager
from src.section_document_processor import SectionDocumentProcessor
from src.unified_llm_manager import UnifiedLLMManager
from src.config import PATH_CONFIG, RETRIEVAL_CONFIG, CHUNKING_CONFIG, OPTIMIZED_Ollama_CONFIG

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot Technique Avancé",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

class TechChatbotApp:
    """Application principale avec le système de retrieval avancé"""
    
    def __init__(self):
        self.doc_processor = None
        self.retrieval_manager = None
        self.llm_manager = None
        self.vector_manager = None
    
    # def init_session_state(self):
    #     """Initialise l'état de la session et les composants avancés"""
    #     # État de base
    #     default_states = {
    #         "messages": [],
    #         "vector_store_loaded": False,
    #         "current_document": None,
    #         "document_processed": False,
    #         "vector_store_path": None,
    #         "components_initialized": False,
    #         "document_stats": None,
    #         "retrieval_ready": False
    #     }
        
    #     for key, value in default_states.items():
    #         if key not in st.session_state:
    #             st.session_state[key] = value
        
    #     # Initialisation des composants (une seule fois)
    #     if not st.session_state.get("components_initialized", False):
    #         self._initialize_components()
    def init_session_state(self):
        """Initialise l'état de la session ET les composants"""
        # État de base
        default_states = {
            "messages": [],
            "vector_store_loaded": False,
            "current_document": None,
            "document_processed": False,
            "vector_store_path": None,
            "components_initialized": False,
            "document_stats": None,
            "retrieval_ready": False
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Initialisation des composants (une seule fois)
        if not st.session_state.get("components_initialized", False):
            try:
                PATH_CONFIG.create_directories()
                
                # ✅ CORRECTION: Initialiser TOUS les composants
                from src.section_document_processor import SectionDocumentProcessor
                from src.retrieval_manager import RetrievalManager
                from src.unified_llm_manager import UnifiedLLMManager
                from src.config import CHUNKING_CONFIG
                
                # 1. Document Processor
                self.doc_processor = SectionDocumentProcessor(
                    min_section_length=CHUNKING_CONFIG.MIN_SECTION_LENGTH,
                    max_section_length=CHUNKING_CONFIG.MAX_SECTION_LENGTH
                )
                
                # 2. Retrieval Manager
                self.retrieval_manager = RetrievalManager()
                
                # 3. LLM Manager
                self.llm_manager = UnifiedLLMManager(
                    mode="auto",
                    ollama_model="mistral"
                )
                
                # Stocker les composants dans session_state
                st.session_state.doc_processor = self.doc_processor
                st.session_state.retrieval_manager = self.retrieval_manager
                st.session_state.llm_manager = self.llm_manager
                st.session_state.components_initialized = True
                
                logger.info("✅ Tous les composants avancés initialisés avec succès")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation: {str(e)}", exc_info=True)
                st.error(f"Erreur lors de l'initialisation: {str(e)}")
        else:
            # Récupérer les composants depuis session_state
            self.doc_processor = st.session_state.doc_processor
            self.retrieval_manager = st.session_state.retrieval_manager
            self.llm_manager = st.session_state.llm_manager
        
        # Recharger la base vectorielle si elle était déjà chargée
        if (st.session_state.vector_store_loaded and 
            st.session_state.vector_store_path):
            try:
                # Note: Le retrieval_manager gère maintenant le vector store
                logger.info("Base de connaissances déjà chargée")
            except Exception as e:
                logger.error(f"Erreur rechargement base: {str(e)}")
                st.session_state.vector_store_loaded = False
    
    def _initialize_components(self):
        """Initialise tous les composants avec gestion d'erreur robuste"""
        try:
            # Créer les répertoires nécessaires
            PATH_CONFIG.create_directories()
            
            # 1. Initialiser le processeur de documents par sections
            self._initialize_document_processor()
            
            # 2. Initialiser le système de retrieval avancé
            self._initialize_retrieval_system()
            
            # 3. Initialiser le gestionnaire de LLM
            self._initialize_llm_manager()
            
            # Stocker les composants dans session_state
            st.session_state.doc_processor = self.doc_processor
            st.session_state.retrieval_manager = self.retrieval_manager
            st.session_state.llm_manager = self.llm_manager
            st.session_state.components_initialized = True
            
            logger.info("✅ Tous les composants avancés initialisés avec succès")
            
        except Exception as e:
            error_msg = f"Erreur lors de l'initialisation des composants: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.session_state.components_initialized = False
    
    def _initialize_document_processor(self):
        """Initialise le processeur de documents par sections"""
        try:
            self.doc_processor = SectionDocumentProcessor(
                min_section_length=CHUNKING_CONFIG.MIN_SECTION_LENGTH,
                max_section_length=CHUNKING_CONFIG.MAX_SECTION_LENGTH
            )
            logger.info("✅ Processeur de documents par sections initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation processeur documents: {e}")
            raise
    
    def _initialize_retrieval_system(self):
        """Initialise le système de retrieval avancé"""
        try:
            self.retrieval_manager = RetrievalManager()
            logger.info("✅ Système de retrieval avancé initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation retrieval system: {e}")
            # Fallback vers un système basique si disponible
            self._initialize_fallback_retrieval()
    
    def _initialize_fallback_retrieval(self):
        """Initialise un système de retrieval de fallback"""
        try:
            from src.advanced_vector_store import AdvancedVectorStore
            from src.config import OPTIMIZED_Ollama_CONFIG
            
            logger.warning("🔄 Utilisation du système de retrieval de fallback")
            self.retrieval_manager = None
            self.vector_manager = AdvancedVectorStore()
        except Exception as e:
            logger.error(f"❌ Fallback retrieval également échoué: {e}")
            raise
    
    def _initialize_llm_manager(self):
        """Initialise le gestionnaire de LLM unifié"""
        try:
            self.llm_manager = UnifiedLLMManager(
                mode="Ollama",  
                ollama_model="mistral"
            )
            
            if self.llm_manager.is_initialized:
                logger.info(f"✅ LLM Manager initialisé - Mode: {self.llm_manager.mode}")
            else:
                logger.warning("⚠️ LLM Manager initialisé mais non opérationnel")
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation LLM Manager: {e}")
            # Créer un manager minimal pour éviter les crashs
            self.llm_manager = self._create_minimal_llm_manager()
    
    def _create_minimal_llm_manager(self):
        """Crée un gestionnaire LLM minimal en cas d'erreur"""
        class MinimalLLMManager:
            def __init__(self):
                self.is_initialized = False
                self.mode = "none"
            
            def generate_response(self, prompt):
                return "❌ Le système de génération n'est pas disponible. Vérifiez la configuration."
            
            def create_technical_prompt(self, context, question):
                return f"Contexte: {context}\nQuestion: {question}\nRéponse:"
            
            def get_model_info(self):
                return {"model": "Aucun", "status": "❌ Erreur", "initialized": False}
            
            def test_generation(self):
                return "❌ Non disponible"
            
            def initialize(self, **kwargs):
                pass
        
        return MinimalLLMManager()
    
    def process_uploaded_file(self, uploaded_file):
        """Traite un fichier uploadé avec le système avancé"""
        try:
            with st.spinner("🔍 Analyse de la structure du document..."):
                # Sauvegarder le fichier temporairement
                file_path = self._save_uploaded_file(uploaded_file)
                
                # Afficher les informations du document
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                st.info(f"**Fichier:** {uploaded_file.name} ({file_size:.2f} MB)")
                
                # Charger et analyser le document
                documents = self.doc_processor.load_pdf(file_path)
                st.info(f"**Pages chargées:** {len(documents)}")
                
                # Analyser la structure du document
                with st.spinner("📊 Analyse de la structure technique..."):
                    stats = self.doc_processor.get_document_statistics(documents)
                    st.session_state.document_stats = stats
                
                # Afficher les statistiques
                self._display_document_stats(stats)
                
                # Traitement par sections techniques
                with st.spinner("✂️ Découpage en sections techniques..."):
                    chunks = self.doc_processor.process_technical_document(documents)
                
                st.info(f"**Sections créées:** {len(chunks)}")
                
                # Indexation avancée
                with st.spinner("🚀 Indexation avancée dans la base de connaissances..."):
                    collection_name = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.retrieval_manager.index_documents(chunks, collection_name)
                
                # Mise à jour de l'état
                st.session_state.vector_store_loaded = True
                st.session_state.current_document = uploaded_file.name
                st.session_state.document_processed = True
                st.session_state.retrieval_ready = True
                
                # Message de succès
                success_msg = f"""
                <div class="success-box">
                <h4>✅ Document traité avec succès !</h4>
                <ul>
                    <li><strong>Document:</strong> {uploaded_file.name}</li>
                    <li><strong>Pages analysées:</strong> {len(documents)}</li>
                    <li><strong>Sections techniques:</strong> {len(chunks)}</li>
                    <li><strong>Méthode:</strong> Découpage par sections</li>
                    <li><strong>Indexation:</strong> Système de retrieval avancé</li>
                </ul>
                </div>
                """
                st.markdown(success_msg, unsafe_allow_html=True)
                
                # Forcer le rerun pour mettre à jour l'interface
                st.rerun()
                
        except Exception as e:
            error_msg = f"Erreur lors du traitement du document: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(f"""
            <div class="warning-box">
            <h4>❌ Erreur de traitement</h4>
            <p>{error_msg}</p>
            <p><strong>Conseils de dépannage:</strong></p>
            <ul>
                <li>Vérifiez que le PDF n'est pas protégé ou corrompu</li>
                <li>Assurez-vous que le document contient du texte (pas seulement des images)</li>
                <li>Essayez avec un document plus petit si possible</li>
            </ul>
            </div>
            """)
    
    def _save_uploaded_file(self, uploaded_file) -> str:
        """Sauvegarde un fichier uploadé avec gestion des erreurs"""
        try:
            os.makedirs(PATH_CONFIG.UPLOAD_DIR, exist_ok=True)
            file_path = os.path.join(PATH_CONFIG.UPLOAD_DIR, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"Fichier sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde fichier: {e}")
            # Fallback: utiliser un fichier temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                return tmp_file.name
    
    def _display_document_stats(self, stats: dict):
        """Affiche les statistiques du document"""
        if not stats:
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pages", stats.get("total_pages", 0))
        
        with col2:
            st.metric("Sections", stats.get("total_chunks", 0))
        
        with col3:
            avg_length = stats.get("avg_chunk_length", 0)
            st.metric("Longueur moyenne", f"{avg_length:.0f} car.")
        
        # Afficher les types de sections
        section_types = stats.get("section_types", {})
        if section_types:
            st.info("**Structure détectée:** " + 
                   " | ".join([f"{k}: {v}" for k, v in section_types.items()]))
    
    def render_chat_interface(self):
        """Affiche l'interface de chat avec le système avancé"""
        st.markdown('<div class="main-header">🤖 Assistant Technique Avancé</div>', unsafe_allow_html=True)
        
        # Vérification du système
        if not self._check_system_ready():
            self._display_system_status()
            return
        
        # Afficher le statut du système
        self._display_active_system_info()
        
        # Zone de chat
        self._render_chat_container()
        
        # Informations détaillées
        self._render_system_details()
    
    def _check_system_ready(self) -> bool:
        """Vérifie si le système est prêt pour les requêtes"""
        # Vérifier l'initialisation des composants
        if not st.session_state.get("components_initialized", False):
            return False
        
        # Vérifier que le retrieval est prêt
        if not st.session_state.get("retrieval_ready", False):
            return False
        
        # Vérifier que le LLM est initialisé
        if not self.llm_manager or not self.llm_manager.is_initialized:
            return False
        
        return True
    
    def _display_system_status(self):
        """Affiche le statut du système"""
        st.markdown("""
        <div style='
            background-color: #fff3cd; 
            border: 1px solid #ffeaa7; 
            border-radius: 0.5rem; 
            padding: 1rem; 
            margin-bottom: 1rem;
        '>
            <h4 style='color: #856404; margin-top: 0;'>⚠️ Système en cours de configuration</h4>
            <p style='color: #856404; margin-bottom: 0;'>
                Pour utiliser l'assistant technique :
            </p>
            <ol style='color: #856404;'>
                <li><strong>Téléchargez</strong> un document PDF dans la sidebar</li>
                <li><strong>Cliquez sur "Traiter le document"</strong></li>
                <li>Ou <strong>chargez une base existante</strong> depuis la liste</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher aussi le statut des composants
        with st.expander("🔧 Statut détaillé du système", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if hasattr(self, 'doc_processor') and self.doc_processor:
                    st.success("✅ Processeur de documents")
                else:
                    st.error("❌ Processeur de documents")
                    
            with col2:
                if hasattr(self, 'retrieval_manager') and self.retrieval_manager:
                    st.success("✅ Système de recherche")
                elif hasattr(self, 'vector_manager') and self.vector_manager:
                    st.warning("⚠️ Système de recherche (fallback)")
                else:
                    st.error("❌ Système de recherche")
                    
            with col3:
                if hasattr(self, 'llm_manager') and self.llm_manager and self.llm_manager.is_initialized:
                    st.success("✅ Modèle de langue")
                else:
                    st.error("❌ Modèle de langue")

        
        # Afficher les informations de debug
        with st.expander("🔧 Informations de débogage système"):
            st.write("**État des composants:**")
            st.write(f"- Composants initialisés: {st.session_state.get('components_initialized', False)}")
            st.write(f"- Retrieval prêt: {st.session_state.get('retrieval_ready', False)}")
            st.write(f"- LLM initialisé: {self.llm_manager.is_initialized if self.llm_manager else False}")
            st.write(f"- Document chargé: {st.session_state.get('document_processed', False)}")
    
    def _display_active_system_info(self):
        """Affiche les informations du système actif"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.current_document:
                st.success(f"**Document:** {st.session_state.current_document}")
            else:
                st.info("**Document:** Aucun")
        
        with col2:
            if self.llm_manager:
                model_info = self.llm_manager.get_model_info()
                status_icon = "✅" if model_info.get("initialized", False) else "❌"
                st.info(f"**Modèle:** {model_info.get('model', 'Inconnu')} {status_icon}")
        
        with col3:
            st.info(f"**Recherche:** {RETRIEVAL_CONFIG.SEARCH_STRATEGY.title()}")
    
    def _render_chat_container(self):
        """Affiche le conteneur de chat"""
        chat_container = st.container()
        
        with chat_container:
            # Affichage de l'historique des messages
            self._render_chat_history()
            
            # Input utilisateur
            self._handle_user_input()
    
    def _render_chat_history(self):
        """Affiche l'historique des messages"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def _handle_user_input(self):
        """Gère l'input utilisateur et génère les réponses"""
        if prompt := st.chat_input("Posez votre question technique..."):
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Générer la réponse
            self._generate_assistant_response(prompt)
    
    def _generate_assistant_response(self, prompt: str):
        """Génère la réponse de l'assistant avec le système avancé"""
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche dans la documentation..."):
                try:
                    # Recherche avancée avec filtres
                    search_filters = self._get_search_filters()
                    similar_docs = self.retrieval_manager.search(
                        query=prompt,
                        k=RETRIEVAL_CONFIG.FINAL_RESULTS,
                        filters=search_filters
                    )
                    
                    if not similar_docs:
                        st.warning("❌ Aucune information pertinente trouvée dans le document.")
                        response = "Je n'ai pas trouvé d'information pertinente dans le document pour répondre à votre question."
                    else:
                        # Afficher les résultats de recherche (optionnel)
                        self._display_search_results(similar_docs)
                        
                        # Préparer le contexte
                        context = self._prepare_context(similar_docs)
                        
                        # Générer la réponse
                        with st.spinner("🤖 Génération de la réponse..."):
                            full_prompt = self.llm_manager.create_technical_prompt(context, prompt)
                            response = self.llm_manager.generate_response(full_prompt)
                    
                    # Afficher la réponse
                    st.markdown(response)
                    
                    # Ajouter à l'historique
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Erreur lors de la génération de la réponse: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error("❌ Une erreur est survenue lors de la génération de la réponse.")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Désolé, une erreur technique m'empêche de répondre pour le moment."
                    })
    
    def _get_search_filters(self) -> dict:
        """Retourne les filtres de recherche appropriés"""
        # Prioriser les sections techniques importantes
        priority_sections = ["api", "installation", "troubleshooting", "configuration", "examples"]
        
        if RETRIEVAL_CONFIG.ENABLE_METADATA_FILTERING:
            return {"section_type": {"$in": priority_sections}}
        
        return None
    
    def _display_search_results(self, similar_docs: list):
        """Affiche les résultats de recherche (optionnel)"""
        if len(similar_docs) > 0 and st.checkbox("🔍 Voir les sections trouvées", False):
            st.info(f"**{len(similar_docs)} sections pertinentes trouvées:**")
            
            for i, doc in enumerate(similar_docs):
                with st.expander(f"Section {i+1}: {doc['metadata'].get('title', 'Sans titre')}"):
                    st.write(f"**Type:** {doc['metadata'].get('section_type', 'content')}")
                    st.write(f"**Score:** {doc.get('final_score', 0):.3f}")
                    st.write(f"**Contenu:** {doc.get('formatted_content', doc['content'])}")
    
    def _prepare_context(self, similar_docs: list) -> str:
        """Prépare le contexte à partir des documents similaires"""
        context_parts = []
        
        for i, doc in enumerate(similar_docs):
            section_info = f"""
## Section {i+1}: {doc['metadata'].get('title', 'Sans titre')}
**Type:** {doc['metadata'].get('section_type', 'content')}
**Pertinence:** {doc.get('final_score', 0):.3f}

{doc['content']}
"""
            context_parts.append(section_info)
        
        return "\n\n".join(context_parts)
    
    def _render_system_details(self):
        """Affiche les détails du système"""
        with st.expander("📊 Détails du système technique"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Configuration Retrieval:**")
                st.write(f"- Stratégie: {RETRIEVAL_CONFIG.SEARCH_STRATEGY}")
                st.write(f"- Modèle embedding: {OPTIMIZED_Ollama_CONFIG.OLLAMA_EMBEDDING_MODEL}")
                st.write(f"- Re-ranking: {'✅' if RETRIEVAL_CONFIG.RERANK_ENABLED else '❌'}")
                st.write(f"- Résultats initiaux: {RETRIEVAL_CONFIG.INITIAL_RESULTS}")
                st.write(f"- Résultats finaux: {RETRIEVAL_CONFIG.FINAL_RESULTS}")
            
            with col2:
                st.write("**Configuration Chunking:**")
                st.write(f"- Stratégie: {CHUNKING_CONFIG.STRATEGY}")
                st.write(f"- Longueur min: {CHUNKING_CONFIG.MIN_SECTION_LENGTH}")
                st.write(f"- Longueur max: {CHUNKING_CONFIG.MAX_SECTION_LENGTH}")
                st.write(f"- Sections prioritaires: {CHUNKING_CONFIG.PRIORITIZE_TECHNICAL_SECTIONS}")
            
            # Test du système
            if st.button("🧪 Tester le système complet"):
                self._run_system_test()
    
    def _run_system_test(self):
        """Exécute un test complet du système"""
        with st.spinner("Exécution des tests système..."):
            test_results = []
            
            # Test du LLM
            llm_test = self.llm_manager.test_generation()
            test_results.append(("LLM", llm_test))
            
            # Test du retrieval (si un document est chargé)
            if st.session_state.retrieval_ready:
                try:
                    retrieval_test = self.retrieval_manager.search("test", k=1)
                    test_results.append(("Retrieval", f"✅ {len(retrieval_test)} résultats"))
                except Exception as e:
                    test_results.append(("Retrieval", f"❌ {str(e)}"))
            else:
                test_results.append(("Retrieval", "❌ Aucun document chargé"))
            
            # Affichage des résultats
            st.write("**Résultats des tests:**")
            for component, result in test_results:
                st.write(f"- {component}: {result}")
    
    def render_sidebar(self):
        """Affiche la sidebar avec les contrôles avancés"""
        with st.sidebar:
            st.title("📁 Gestion des Documents")
            
            # Upload de document
            uploaded_file = st.file_uploader(
                "Téléchargez un document technique (PDF)",
                type="pdf",
                help="Documentation technique, manuels, spécifications..."
            )
            
            if uploaded_file is not None:
                if st.button("🚀 Traiter le document", type="primary", use_container_width=True):
                    self.process_uploaded_file(uploaded_file)
            
            # Informations système
            self._render_system_info()
            
            # Contrôles avancés
            self._render_advanced_controls()
    
    def _render_system_info(self):
        """Affiche les informations système dans la sidebar"""
        st.subheader("💻 Informations Système")
        
        # Informations LLM
        if self.llm_manager:
            model_info = self.llm_manager.get_model_info()
            st.write(f"**Modèle:** {model_info.get('model', 'Inconnu')}")
        
        # Informations retrieval
        if st.session_state.retrieval_ready:
            st.success("✅ Base de connaissances active")
            if st.session_state.document_stats:
                stats = st.session_state.document_stats
                st.write(f"**Sections:** {stats.get('total_chunks', 0)}")
        else:
            st.warning("❌ Aucune base chargée")
    
    def _render_advanced_controls(self):
        """Affiche les contrôles avancés"""
        st.subheader("⚙️ Contrôles Avancés")
        
        # Réinitialisation
        if st.button("🔄 Réinitialiser la conversation"):
            st.session_state.messages = []
            st.rerun()
        
        # Debug
        if st.button("🐛 Mode Debug"):
            self._toggle_debug_mode()
    
    def _toggle_debug_mode(self):
        """Active/désactive le mode debug"""
        debug_mode = not st.session_state.get("debug_mode", False)
        st.session_state.debug_mode = debug_mode
        
        if debug_mode:
            st.info("🔧 Mode debug activé - Vérifiez la console pour les logs détaillés")
        else:
            st.info("🔧 Mode debug désactivé")
    
    def run(self):
        """Lance l'application"""
        load_css()
        self.init_session_state()
        self.render_sidebar()
        self.render_chat_interface()

# Lancement de l'application
if __name__ == "__main__":
    app = TechChatbotApp()
    app.run()