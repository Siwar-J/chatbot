# app.py
import streamlit as st
import os
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import de nos modules
from src.config import MODEL_CONFIG, PROCESSING_CONFIG, PATH_CONFIG
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_manager import LLMManager

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chatbot Technique",
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
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f0f0f0;
        border-left: 4px solid #ff7f0e;
    }
    .document-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)

class TechChatbotApp:
    """Application principale du chatbot technique"""
    
    def __init__(self):
        # Initialisation différée dans init_session_state
        self.doc_processor = None
        self.vector_manager = None
        self.llm_manager = None
    
    def init_session_state(self):
        """Initialise l'état de la session ET les composants"""
        # État de base
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "vector_store_loaded" not in st.session_state:
            st.session_state.vector_store_loaded = False
        if "current_document" not in st.session_state:
            st.session_state.current_document = None
        if "document_processed" not in st.session_state:
            st.session_state.document_processed = False
        if "vector_store_path" not in st.session_state:
            st.session_state.vector_store_path = None
        if "components_initialized" not in st.session_state:
            st.session_state.components_initialized = False
        
        # Initialisation des composants (une seule fois)
        if not st.session_state.get("components_initialized", False):
            try:
                PATH_CONFIG.create_directories()
                self.doc_processor = DocumentProcessor(
                    chunk_size=PROCESSING_CONFIG.CHUNK_SIZE,
                    chunk_overlap=PROCESSING_CONFIG.CHUNK_OVERLAP
                )
                self.vector_manager = VectorStoreManager(
                    embedding_model=MODEL_CONFIG.EMBEDDING_MODEL
                )
                self.llm_manager = LLMManager(
                    model_name=MODEL_CONFIG.LLM_MODEL
                )
                
                # Stocker les composants dans session_state
                st.session_state.doc_processor = self.doc_processor
                st.session_state.vector_manager = self.vector_manager
                st.session_state.llm_manager = self.llm_manager
                st.session_state.components_initialized = True
                
                logger.info("Composants initialisés avec succès")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation: {str(e)}")
                st.error(f"Erreur lors de l'initialisation: {str(e)}")
        else:
            # Récupérer les composants depuis session_state
            self.doc_processor = st.session_state.doc_processor
            self.vector_manager = st.session_state.vector_manager
            self.llm_manager = st.session_state.llm_manager
        
        # Recharger la base vectorielle si elle était déjà chargée
        if (st.session_state.vector_store_loaded and 
            st.session_state.vector_store_path and 
            self.vector_manager and not self.vector_manager.vector_store):
            try:
                self.vector_manager.load_vector_store(st.session_state.vector_store_path)
                logger.info(f"Base vectorielle rechargée: {st.session_state.vector_store_path}")
            except Exception as e:
                logger.error(f"Erreur rechargement base: {str(e)}")
                st.session_state.vector_store_loaded = False
    
    def get_available_vector_stores(self) -> list:
        """Retourne la liste des bases vectorielles disponibles"""
        try:
            if os.path.exists(PATH_CONFIG.VECTOR_STORE_DIR):
                stores = [d for d in os.listdir(PATH_CONFIG.VECTOR_STORE_DIR) 
                         if os.path.isdir(os.path.join(PATH_CONFIG.VECTOR_STORE_DIR, d))]
                logger.info(f"Bases vectorielles disponibles: {stores}")
                return stores
            return []
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des bases: {str(e)}")
            return []
    
    def process_uploaded_file(self, uploaded_file):
        """Traite un fichier uploadé"""
        try:
            with st.spinner("Traitement du document en cours..."):
                # Sauvegarde du fichier
                file_path = self.doc_processor.save_uploaded_file(
                    uploaded_file, PATH_CONFIG.UPLOAD_DIR
                )
                
                # Chargement et traitement du PDF
                documents = self.doc_processor.load_pdf(file_path)
                chunks = self.doc_processor.process_documents(documents)
                
                # Création de la base vectorielle
                store_name = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                persist_path = os.path.join(PATH_CONFIG.VECTOR_STORE_DIR, store_name)
                
                self.vector_manager.create_vector_store(chunks, persist_path)
                
                # Initialisation du LLM si pas déjà fait
                if not self.llm_manager.is_initialized:
                    self.llm_manager.initialize(
                        max_new_tokens=MODEL_CONFIG.MAX_NEW_TOKENS,
                        temperature=MODEL_CONFIG.TEMPERATURE,
                        top_p=MODEL_CONFIG.TOP_P,
                        repetition_penalty=MODEL_CONFIG.REPETITION_PENALTY
                    )
                
                # Mise à jour de l'état
                st.session_state.vector_store_loaded = True
                st.session_state.current_document = uploaded_file.name
                st.session_state.document_processed = True
                st.session_state.vector_store_path = persist_path
                
                # Mettre à jour le vector_manager dans session_state
                st.session_state.vector_manager = self.vector_manager
                st.session_state.llm_manager = self.llm_manager
                
                logger.info(f"Document '{uploaded_file.name}' traité avec succès, base créée: {persist_path}")
                st.success(f"✅ Document '{uploaded_file.name}' traité avec succès!")
                
                # Forcer le rerun pour mettre à jour l'interface
                st.rerun()
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement: {str(e)}", exc_info=True)
            st.error(f"Erreur lors du traitement du document: {str(e)}")
    
    def load_existing_vector_store(self, store_name: str):
        """Charge une base vectorielle existante"""
        try:
            with st.spinner("Chargement de la base..."):
                persist_path = os.path.join(PATH_CONFIG.VECTOR_STORE_DIR, store_name)
                
                # Vérification que le chemin existe
                if not os.path.exists(persist_path):
                    st.error(f"Base vectorielle non trouvée: {persist_path}")
                    return
                
                # Chargement de la base vectorielle
                self.vector_manager.load_vector_store(persist_path)
                
                # Initialisation du LLM si pas déjà fait
                if not self.llm_manager.is_initialized:
                    self.llm_manager.initialize(
                        max_new_tokens=MODEL_CONFIG.MAX_NEW_TOKENS,
                        temperature=MODEL_CONFIG.TEMPERATURE,
                        top_p=MODEL_CONFIG.TOP_P,
                        repetition_penalty=MODEL_CONFIG.REPETITION_PENALTY
                    )
                
                # Mise à jour de l'état
                st.session_state.vector_store_loaded = True
                st.session_state.current_document = store_name
                st.session_state.vector_store_path = persist_path
                st.session_state.vector_manager = self.vector_manager
                st.session_state.llm_manager = self.llm_manager
                
                logger.info(f"Base vectorielle chargée: {persist_path}")
                st.success(f"✅ Base '{store_name}' chargée avec succès!")
                st.rerun()
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {str(e)}")
            st.error(f"Erreur lors du chargement de la base: {str(e)}")
    
    def check_system_ready(self) -> bool:
        """Vérifie si le système est prêt pour répondre aux questions"""
        if not st.session_state.vector_store_loaded:
            return False
        
        if not self.vector_manager or not self.vector_manager.vector_store:
            return False
        
        if not self.llm_manager or not self.llm_manager.is_initialized:
            return False
        
        return True
    
    def render_model_info(self):
        """Affiche les informations du modèle"""
        with st.sidebar:
            if self.llm_manager and hasattr(self.llm_manager, 'is_initialized'):
                st.subheader("🧠 Modèle")
                if self.llm_manager.is_initialized:
                    model_info = self.llm_manager.get_model_info()
                    st.success(f"**Modèle:** {model_info['model'].split('/')[-1]}")
                    st.success(f"**Device:** {model_info['device'].upper()}")
                    st.success(f"**Statut:** ✅ Initialisé")
                    
                    # Test rapide
                    if st.button("🧪 Tester le modèle"):
                        with st.spinner("Test en cours..."):
                            test_result = self.llm_manager.test_generation()
                            st.info(f"**Résultat:** {test_result}")
                else:
                    st.warning("**Statut:** ❌ Non initialisé")
    
    def render_sidebar(self):
        """Affiche la sidebar avec les contrôles"""
        with st.sidebar:
            st.title("📁 Gestion des Documents")
            
            # Upload de document
            uploaded_file = st.file_uploader(
                "Téléchargez un document technique (PDF)",
                type="pdf",
                help="Seuls les fichiers PDF sont supportés"
            )
            
            if uploaded_file is not None:
                if st.button("Traiter le document", type="primary"):
                    self.process_uploaded_file(uploaded_file)
            
            # Chargement d'une base existante
            st.subheader("Base de connaissances")
            vector_stores = self.get_available_vector_stores()
            
            if vector_stores:
                selected_store = st.selectbox(
                    "Charger une base existante",
                    options=vector_stores,
                    format_func=lambda x: x.replace("_", " ").title()
                )
                
                if st.button("Charger la base sélectionnée"):
                    self.load_existing_vector_store(selected_store)
            else:
                st.info("Aucune base vectorielle disponible. Téléchargez d'abord un document PDF.")
            
            # Informations système - CORRECTION: Vérifier que vector_manager existe
            st.subheader("💻 Informations Système")
            if self.vector_manager and hasattr(self.vector_manager, 'device'):
                device_display = "GPU 🚀" if self.vector_manager.device == "cuda" else "CPU ⚡"
                st.write(f"**Device:** {device_display}")
            else:
                st.write(f"**Device:** CPU ⚡ (par défaut)")
            
            self.render_model_info()
            
            if st.session_state.vector_store_loaded:
                if self.vector_manager:
                    collection_info = self.vector_manager.get_collection_info()
                    st.success(f"✅ Base chargée ({collection_info['count']} documents)")
                    
                    # Test de fonctionnement
                    try:
                        test_results = self.vector_manager.similarity_search("test", k=1)
                        st.success(f"🔍 Recherche test: OK ({len(test_results)} résultat)")
                    except Exception as e:
                        st.error(f"🔍 Recherche test: ÉCHEC - {e}")
            else:
                st.warning("❌ Aucune base vectorielle chargée")
            
            # Paramètres
            st.subheader("⚙️ Paramètres")
            st.slider(
                "Nombre de résultats similaires",
                min_value=1,
                max_value=10,
                value=PROCESSING_CONFIG.SIMILARITY_TOP_K,
                key="similarity_top_k"
            )
            
            # Bouton de réinitialisation
            if st.button("🔄 Réinitialiser la conversation"):
                st.session_state.messages = []
                st.rerun()
            
            # Bouton de debug
            if st.button("🐛 Debug System"):
                self.render_debug_info()
    
    def render_debug_info(self):
        """Affiche les informations de debug"""
        with st.expander("🐛 Debug Information", expanded=True):
            st.write("### État du système")
            st.write(f"- Vector store loaded: {st.session_state.vector_store_loaded}")
            st.write(f"- Vector manager exists: {self.vector_manager is not None}")
            if self.vector_manager:
                st.write(f"- Vector store instance: {self.vector_manager.vector_store is not None}")
            st.write(f"- LLM initialized: {self.llm_manager.is_initialized if self.llm_manager else False}")
            st.write(f"- Vector store path: {st.session_state.vector_store_path}")
            
            if self.vector_manager and self.vector_manager.vector_store:
                try:
                    collection_info = self.vector_manager.get_collection_info()
                    st.write(f"- Document count: {collection_info['count']}")
                    
                    # Test de recherche
                    test_query = "programming"
                    test_results = self.vector_manager.similarity_search(test_query, k=2)
                    st.write(f"- Test search '{test_query}': {len(test_results)} résultats")
                    
                    for i, doc in enumerate(test_results):
                        st.write(f"  Résultat {i+1}: {doc.page_content[:80]}...")
                        
                except Exception as e:
                    st.error(f"Erreur lors du test: {e}")
    
    def render_chat_interface(self):
        """Affiche l'interface de chat"""
        st.markdown('<div class="main-header">🤖 Chatbot Technique</div>', unsafe_allow_html=True)
        
        # Avertissement si aucune base n'est chargée
        if not self.check_system_ready():
            st.warning("""
            **⚠️ Aucune base de connaissances chargée**
            
            Pour utiliser le chatbot :
            1. **Téléchargez** un document PDF dans la sidebar
            2. **Cliquez sur "Traiter le document"**
            OU
            1. **Chargez une base existante** depuis la liste déroulante
            """)
        else:
            # Afficher les informations de la base chargée
            if self.vector_manager:
                collection_info = self.vector_manager.get_collection_info()
                st.success(f"**✅ Base de connaissances chargée :** {collection_info['count']} documents disponibles")
        
        # Zone de chat
        chat_container = st.container()
        
        with chat_container:
            # Affichage des messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Input utilisateur
            if prompt := st.chat_input("Posez votre question technique..."):
                # Vérification qu'une base est chargée
                if not self.check_system_ready():
                    st.error("❌ Veuillez d'abord charger ou traiter un document.")
                    return
                
                # Vérification finale
                if not self.vector_manager.vector_store:
                    st.error("❌ Base vectorielle non disponible. Veuillez recharger le document.")
                    return
                
                # Ajout du message utilisateur
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Génération de la réponse
                with st.chat_message("assistant"):
                    with st.spinner("Recherche dans les documents..."):
                        try:
                            # Recherche de documents similaires
                            similar_docs = self.vector_manager.similarity_search(
                                prompt, k=st.session_state.get("similarity_top_k", 4)
                            )
                            
                            # Création du contexte
                            context = "\n\n".join([
                                f"**Extrait {i+1}:** {doc.page_content}" 
                                for i, doc in enumerate(similar_docs)
                            ])
                            
                            # Création du prompt
                            full_prompt = self.llm_manager.create_technical_prompt(context, prompt)
                            
                            # Génération de la réponse
                            response = self.llm_manager.generate_response(full_prompt)
                            
                            # Affichage de la réponse
                            st.markdown(response)
                            
                            # Ajout à l'historique
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                            
                        except Exception as e:
                            error_msg = f"Erreur lors de la génération: {str(e)}"
                            logger.error(f"Erreur détaillée: {str(e)}", exc_info=True)
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
        
        # Informations sur le document courant
        if st.session_state.vector_store_loaded:
            with st.expander("📊 Informations du document courant"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Document:** {st.session_state.current_document}")
                with col2:
                    if self.vector_manager:
                        collection_info = self.vector_manager.get_collection_info()
                        st.write(f"**Documents indexés:** {collection_info['count']}")
                with col3:
                    if st.session_state.vector_store_path:
                        st.write(f"**Chemin base:** {os.path.basename(st.session_state.vector_store_path)}")
    
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