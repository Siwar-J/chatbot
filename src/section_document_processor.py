# src/section_document_processor.py
import os
import logging
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pypdf

logger = logging.getLogger(__name__)

class SectionDocumentProcessor:
    """Processeur de documents avec chunking par sections techniques"""
    
    def __init__(self, min_section_length: int = 200, max_section_length: int = 1500):
        self.min_section_length = min_section_length
        self.max_section_length = max_section_length
        
        # Patterns pour détecter les sections techniques
        self.section_patterns = [
            r'^\d+\.\s+.+',                    # 1. Titre de section
            r'^\d+\.\d+\.\s+.+',              # 1.1. Sous-section
            r'^[A-Z][A-Za-z\s]+:',            # SECTION: format
            r'^##?\s+.+',                     # ## Titre Markdown
            r'^[IVX]+\.\s+.+',                # I. Titre romain
            r'^[A-Z\s]{3,}$',                 # TITRE EN MAJUSCULES
            r'^\*{3,}.+\*{3,}$',             # ***TITRE***
        ]
        
        logger.info("✅ Processeur par sections initialisé")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Charge un PDF avec extraction améliorée préservant la structure
        """
        try:
            logger.info(f"Chargement du PDF: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
            
            documents = []
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        
                        if text and len(text.strip()) > 100:
                            # Créer un document par page pour préserver la structure
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "source": file_path,
                                    "page": page_num + 1,
                                    "total_pages": len(pdf_reader.pages),
                                    "type": "page"
                                }
                            )
                            documents.append(doc)
                            logger.debug(f"Page {page_num + 1}: {len(text)} caractères")
                            
                    except Exception as e:
                        logger.warning(f"Erreur page {page_num + 1}: {e}")
                        continue
            
            if not documents:
                raise ValueError("Aucun texte valide extrait du PDF")
            
            logger.info(f"✅ PDF chargé: {len(documents)} pages avec contenu")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement PDF: {e}")
            raise
    
    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Détecte les sections dans le texte
        """
        sections = []
        lines = text.split('\n')
        current_section = []
        current_title = "Introduction"
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Vérifier si la ligne est un titre de section
            is_section = any(re.match(pattern, line) for pattern in self.section_patterns)
            
            if is_section and len(line) < 100:  # Éviter les faux positifs
                # Sauvegarder la section précédente
                if current_section and len(' '.join(current_section)) >= self.min_section_length:
                    sections.append({
                        'title': current_title,
                        'content': ' '.join(current_section),
                        'line_start': i - len(current_section)
                    })
                
                # Commencer une nouvelle section
                current_title = line
                current_section = []
            else:
                # Ajouter à la section courante
                if line and len(line) > 10:  # Ignorer les lignes trop courtes
                    current_section.append(line)
        
        # Ajouter la dernière section
        if current_section and len(' '.join(current_section)) >= self.min_section_length:
            sections.append({
                'title': current_title,
                'content': ' '.join(current_section),
                'length': len(' '.join(current_section))
            })
        
        return sections
    
    def process_documents_by_sections(self, documents: List[Document]) -> List[Document]:
        """
        Traite les documents en les découpant par sections naturelles
        """
        try:
            if not documents:
                raise ValueError("Aucun document à traiter")
            
            logger.info(f"Traitement par sections de {len(documents)} documents...")
            
            all_chunks = []
            
            for doc_idx, doc in enumerate(documents):
                text = doc.page_content
                sections = self.detect_sections(text)
                
                logger.info(f"📑 Page {doc_idx + 1}: {len(sections)} sections détectées")
                
                for section_idx, section in enumerate(sections):
                    # Vérifier la longueur de la section
                    content = section['content']
                    if len(content) > self.max_section_length:
                        # Découper les sections trop longues
                        sub_sections = self._split_large_section(content, section['title'])
                        for sub_idx, sub_content in enumerate(sub_sections):
                            chunk = self._create_chunk(
                                content=sub_content,
                                title=f"{section['title']} - Partie {sub_idx + 1}",
                                source_doc=doc,
                                chunk_id=f"{doc_idx+1}_{section_idx+1}_{sub_idx+1}"
                            )
                            all_chunks.append(chunk)
                    else:
                        # Créer un chunk pour la section complète
                        chunk = self._create_chunk(
                            content=content,
                            title=section['title'],
                            source_doc=doc,
                            chunk_id=f"{doc_idx+1}_{section_idx+1}"
                        )
                        all_chunks.append(chunk)
            
            logger.info(f"✅ Traitement terminé: {len(all_chunks)} chunks créés")
            
            # Log des sections créées
            for i, chunk in enumerate(all_chunks[:5]):
                logger.info(f"Section {i+1}: '{chunk.metadata.get('title', 'Sans titre')}' - {len(chunk.page_content)} caractères")
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement par sections: {e}")
            raise
    
    def _split_large_section(self, content: str, title: str) -> List[str]:
        """
        Découpe une section trop longue en sous-sections
        """
        sentences = re.split(r'[.!?]+', content)
        sub_sections = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.max_section_length and current_chunk:
                # Sauvegarder le chunk actuel
                sub_sections.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Ajouter le dernier chunk
        if current_chunk:
            sub_sections.append(' '.join(current_chunk))
        
        return sub_sections
    
    def _create_chunk(self, content: str, title: str, source_doc: Document, chunk_id: str) -> Document:
        """Crée un document chunk avec métadonnées enrichies"""
        return Document(
            page_content=content.strip(),
            metadata={
                **source_doc.metadata,
                "chunk_id": chunk_id,
                "title": title,
                "section_type": self._classify_section_type(title),
                "content_length": len(content),
                "processing_method": "section_based"
            }
        )
    
    def _classify_section_type(self, title: str) -> str:
        """Classifie le type de section"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['introduction', 'résumé', 'abstract']):
            return "introduction"
        elif any(word in title_lower for word in ['installation', 'setup', 'configuration']):
            return "installation"
        elif any(word in title_lower for word in ['api', 'endpoint', 'interface']):
            return "api"
        elif any(word in title_lower for word in ['exemple', 'example', 'sample']):
            return "examples"
        elif any(word in title_lower for word in ['erreur', 'error', 'troubleshooting']):
            return "troubleshooting"
        elif any(word in title_lower for word in ['référence', 'reference']):
            return "reference"
        elif any(word in title_lower for word in ['conclusion', 'summary']):
            return "conclusion"
        else:
            return "content"
    
    def process_technical_document(self, documents: List[Document]) -> List[Document]:
        """
        Méthode spécialisée pour la documentation technique
        """
        try:
            logger.info("🔧 Traitement spécialisé documentation technique...")
            
            chunks = self.process_documents_by_sections(documents)
            
            # Filtrer et organiser les sections techniques
            technical_chunks = []
            for chunk in chunks:
                section_type = chunk.metadata.get('section_type', 'content')
                
                # Prioriser certains types de sections
                priority_sections = ['installation', 'api', 'troubleshooting', 'examples']
                if section_type in priority_sections:
                    technical_chunks.insert(0, chunk)  # Mettre en priorité
                else:
                    technical_chunks.append(chunk)
            
            logger.info(f"✅ Documentation technique traitée: {len(technical_chunks)} chunks organisés")
            return technical_chunks
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement technique: {e}")
            return self.process_documents_by_sections(documents)  # Fallback
    
    def save_uploaded_file(self, uploaded_file, upload_dir: str) -> str:
        """Sauvegarde un fichier uploadé"""
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Fichier sauvegardé: {file_path}")
        return file_path
    
    def get_document_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """Retourne des statistiques sur le document"""
        if not documents:
            return {}
        
        chunks = self.process_documents_by_sections(documents)
        
        section_types = {}
        for chunk in chunks:
            section_type = chunk.metadata.get('section_type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        return {
            "total_pages": len(documents),
            "total_chunks": len(chunks),
            "section_types": section_types,
            "avg_chunk_length": sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
        }