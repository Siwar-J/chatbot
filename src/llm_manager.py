# src/llm_manager.py
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
from langchain_community.llms import HuggingFacePipeline
import gc

logger = logging.getLogger(__name__)

class LLMManager:
    """Gère les modèles légers pour 4GB GPU"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.llm = None
        self.device = "cpu"
        self.is_initialized = False
        
    def initialize(self, **generation_kwargs):
        """
        Initialisation avec configuration corrigée pour éviter la répétition
        """
        try:
            logger.info(f"🚀 Initialisation de {self.model_name} sur {self.device}")
            
            self._clean_memory()
            
            # Configuration SPÉCIFIQUE pour DialoGPT
            safe_kwargs = {
                "max_new_tokens": 150,  # Réduit pour éviter les répétitions
                "temperature": 0.8,     # Augmenté pour plus de créativité
                "top_p": 0.9,
                "top_k": 50,           # Ajouté pour DialoGPT
                "do_sample": True,
                "repetition_penalty": 1.2,  # Augmenté pour éviter les répétitions
                "pad_token_id": 50256,  # IMPORTANT: token de padding pour GPT
                "eos_token_id": 50256,  # Token de fin pour DialoGPT
            }
            safe_kwargs.update(generation_kwargs)
            
            logger.info("📥 Chargement du tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configuration CRITIQUE pour DialoGPT
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"  # Important pour la génération
            
            logger.info("📥 Chargement du modèle sur CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=None,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            logger.info("⚙️ Création du pipeline avec configuration DialoGPT...")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=safe_kwargs["max_new_tokens"],
                temperature=safe_kwargs["temperature"],
                top_p=safe_kwargs["top_p"],
                top_k=safe_kwargs["top_k"],
                repetition_penalty=safe_kwargs["repetition_penalty"],
                do_sample=safe_kwargs["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                device=-1,
                return_full_text=False,  # CRITIQUE: ne pas répéter le prompt
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.is_initialized = True
            
            logger.info(f"✅ {self.model_name} initialisé avec succès")
            
            # Test avec un prompt qui force une réponse différente
            test_prompt = "Utilisateur: Bonjour\nAssistant:"
            logger.info(f"🧪 Test avec prompt: {test_prompt}")
            test_response = self.generate_response(test_prompt)
            logger.info(f"🧪 Réponse de test: {test_response}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}", exc_info=True)
            self._try_alternative_model(**generation_kwargs)
    
    def _try_alternative_model(self, **generation_kwargs):
        """Tente un modèle alternatif si DialoGPT ne fonctionne pas"""
        alternative_models = [
            "microsoft/DialoGPT-large",  # Peut-être mieux configuré
            "distilgpt2",                # Plus basique mais fiable
            "gpt2",                      # Standard
        ]
        
        for alt_model in alternative_models:
            try:
                logger.info(f"🔄 Tentative avec modèle alternatif: {alt_model}")
                self.model_name = alt_model
                self.llm = None
                self.is_initialized = False
                
                self.initialize(**generation_kwargs)
                if self.is_initialized:
                    logger.info(f"✅ Modèle alternatif chargé: {alt_model}")
                    return
                    
            except Exception as e:
                logger.warning(f"❌ {alt_model} a échoué: {e}")
                continue
        
        logger.error("❌ Aucun modèle n'a pu être chargé")
        self.is_initialized = False
    
    def _clean_memory(self):
        """Nettoie la mémoire"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate_response(self, prompt: str) -> str:
        """Génère une réponse avec gestion des répétitions"""
        if not self.is_initialized:
            return "❌ Le modèle n'est pas initialisé"
        
        try:
            logger.info(f"🎯 Génération pour prompt: {prompt[:100]}...")
            
            # Essayer invoke() d'abord
            response = self.llm.invoke(prompt)
            
            logger.info(f"📝 Réponse brute: {response}")
            
            cleaned_response = self._clean_response(response, prompt)
            logger.info(f"✅ Réponse nettoyée: {cleaned_response[:100]}...")
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"❌ Erreur de génération: {str(e)}", exc_info=True)
            return f"❌ Erreur: {str(e)}"
    
    def _clean_response(self, response, original_prompt):
        """Nettoie la réponse en supprimant les répétitions du prompt"""
        if isinstance(response, str):
            text = response
        else:
            text = str(response)
        
        logger.info(f"🧹 Nettoyage - Réponse originale: {text}")
        
        # Supprimer le prompt s'il est répété
        if original_prompt in text:
            text = text.replace(original_prompt, "").strip()
        
        # Supprimer les préfixes communs
        prefixes_to_remove = [
            "Utilisateur:", "User:", "Assistant:", "Bot:",
            "Human:", "AI:", "###", "**"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Nettoyer les espaces multiples
        text = ' '.join(text.split())
        
        # Si la réponse est vide après nettoyage, retourner un message par défaut
        if not text or len(text) < 5:
            text = "Je n'ai pas pu générer une réponse appropriée. Voici les extraits pertinents du document."
        
        logger.info(f"🧹 Réponse nettoyée: {text}")
        return text
    
    def create_technical_prompt(self, context: str, question: str, section_info: str = "") -> str:
        """Prompt optimisé pour la documentation technique structurée"""
        logger.info(f"📝 Prompt créé - Format: conversation DialoGPT")
        return f"""En tant qu'expert technique, analyse les sections de documentation suivantes et réponds à la question.

        STRUCTURE DU DOCUMENT:
        {section_info}

        CONTENU TECHNIQUE:
        {context}

        QUESTION:
        {question}

        INSTRUCTIONS:
        1. Réponds en français de manière technique et structurée
        2. Utilise UNIQUEMENT les informations des sections fournies
        3. Mentionne la section pertinente quand c'est possible
        4. Si l'information est incomplète, indique quelles sections consulter
        5. Pour les questions d'installation/configuration, sois très précis
        6. Pour les erreurs, propose des solutions étape par étape

        RÉPONSE TECHNIQUE DÉTAILLÉE:"""
        
    
    def test_generation(self) -> str:
        """Teste la génération avec différents prompts"""
        if not self.is_initialized:
            return "❌ Modèle non initialisé"
        
        try:
            # Test 1: Prompt simple
            test1 = self.generate_response("Utilisateur: Bonjour\nAssistant:")
            result1 = f"Test1: {test1[:50]}..." if test1 else "Échec"
            
            # Test 2: Prompt avec contexte
            test2_prompt = self.create_technical_prompt(
                "L'IA transforme l'industrie.", 
                "Quel est l'impact de l'IA?"
            )
            test2 = self.generate_response(test2_prompt)
            result2 = f"Test2: {test2[:50]}..." if test2 else "Échec"
            
            return f"✅ Tests: {result1} | {result2}"
            
        except Exception as e:
            return f"❌ Test échoué: {e}"
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle"""
        return {
            "model": self.model_name,
            "device": self.device,
            "initialized": self.is_initialized
        }