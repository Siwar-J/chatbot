# src/document_processor.py
import os
import logging
from typing import List

# Gestion des imports compatibles
try:
    # Nouveaux imports LangChain
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    logger = logging.getLogger(__name__)
except ImportError:
    try:
        # Anciens imports LangChain
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        logger = logging.getLogger(__name__)
    except ImportError as e:
        raise ImportError(
            "Impossible de charger les modules LangChain. "
            "Assurez-vous d'avoir installé les dépendances: "
            "pip install langchain-core langchain-text-splitters"
        )

from langchain_community.document_loaders import PyPDFLoader
import pypdf

class DocumentProcessor:
    """Gère le chargement et le traitement des documents PDF"""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Charge un fichier PDF et retourne les documents
        
        Args:
            file_path: Chemin vers le fichier PDF
            
        Returns:
            Liste de documents LangChain
        """
        try:
            logger.info(f"Chargement du PDF: {file_path}")
            
            # Vérification que le fichier existe et n'est pas vide
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"Le fichier {file_path} est vide")
            
            logger.info(f"Taille du fichier: {file_size} bytes")
            
            # Test de lecture du PDF
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    num_pages = len(pdf_reader.pages)
                    logger.info(f"PDF contient {num_pages} pages (test pypdf)")
                    
                    # Test de lecture du premier page
                    if num_pages > 0:
                        first_page = pdf_reader.pages[0]
                        first_page_text = first_page.extract_text()
                        logger.info(f"Première page - {len(first_page_text)} caractères: {first_page_text[:100]}...")
            except Exception as e:
                logger.warning(f"Test pypdf échoué: {e}")
            
            # Chargement avec LangChain
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("Aucun document chargé - le PDF est peut-être vide ou corrompu")
            
            # Log détaillé du contenu
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(f"PDF chargé avec {len(documents)} pages, {total_chars} caractères au total")
            
            for i, doc in enumerate(documents[:3]):  # Log des 3 premières pages
                content_preview = doc.page_content.replace('\n', ' ')[:150]
                logger.info(f"Page {i+1}: {content_preview}...")
            
            return documents
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du PDF {file_path}: {str(e)}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe les documents en chunks
        
        Args:
            documents: Liste de documents à traiter
            
        Returns:
            Liste de chunks de documents
        """
        try:
            if not documents:
                raise ValueError("Aucun document à traiter")
            
            logger.info(f"Découpage de {len(documents)} documents en chunks...")
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("Aucun chunk généré après le découpage")
            
            logger.info(f"Documents découpés en {len(chunks)} chunks")
            
            # Log des premiers chunks
            for i, chunk in enumerate(chunks[:2]):
                content_preview = chunk.page_content.replace('\n', ' ')[:100]
                logger.info(f"Chunk {i+1}: {content_preview}...")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur lors du découpage des documents: {str(e)}")
            raise
    
    def save_uploaded_file(self, uploaded_file, upload_dir: str) -> str:
        """
        Sauvegarde un fichier uploadé via Streamlit
        
        Args:
            uploaded_file: Fichier uploadé via st.file_uploader
            upload_dir: Répertoire de sauvegarde
            
        Returns:
            Chemin vers le fichier sauvegardé
        """
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Fichier sauvegardé: {file_path} ({os.path.getsize(file_path)} bytes)")
        return file_path

    def get_document_info(self, documents: List[Document]) -> dict:
        """
        Retourne des informations sur les documents
        
        Args:
            documents: Liste de documents
            
        Returns:
            Dict avec les informations
        """
        total_pages = len(documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        return {
            "total_pages": total_pages,
            "total_characters": total_chars,
            "average_chars_per_page": total_chars // total_pages if total_pages > 0 else 0
        }