# 🤖 Chatbot - RAG avec LLMs

Un chatbot intelligent qui répond aux questions en se basant sur vos documents PDF, utilisant la technologie RAG (Retrieval Augmented Generation) et des modèles de langue avancés.

## 📋 Table des Matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Développement](#-développement)
- [Configuration](#-configuration)
- [Dépannage](#-dépannage)

## 🚀 Fonctionnalités

- **📄 Traitement de documents PDF** - Extraction et analyse de contenu du document
- **🔍 Recherche sémantique** - Trouve les informations les plus pertinentes
- **🤖 Génération de réponses** - Réponses précises basées sur le contexte
- **☁️ Support backend** - Ollama
- **💬 Interface intuitive** - Application Streamlit moderne et responsive
- **🔄 Gestion de contexte** - Maintient l'historique des conversations

## 🏗️ Architecture

┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Interface     │    │   Traitement     │    │   Base de      │
│   Streamlit     │◄──►│   des Documents  │◄──►│   Connaissances│
│                 │    │                  │    │   Vectorielle  │
└─────────────────┘    └──────────────────┘    └────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   Gestionnaire  │    │   Embeddings     │    │   Recherche    │
│      LLM        │    │   Sémantiques    │    │   Vectorielle  │
│                 │    │                  │    │                │
└─────────────────┘    └──────────────────┘    └────────────────┘


## 🛠️ Installation

### Prérequis

- Python 3.8+
- Ollama
- 8GB+ de RAM (16GB recommandé pour les gros modèles)

### Installation Pas à Pas

#### 1. Cloner le repository

```bash
git clone git@github.com:Siwar-J/chatbot.git
cd chatbot
```

#### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate    # Windows
```

#### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 4. Configuration des Modèles de Langue

**Option A: Ollama (Recommandé - Local et Performant)**

```bash
# Installer Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger un modèle
ollama pull mistral
```

#### 5. Structure des Dossiers

```bash
tech_chatbot/
├── src/                    # Code source
├── data/
│   ├── uploaded_docs/      # PDFs uploadés
│   └── vector_stores/      # Bases vectorielles
├── static/                 # Assets statiques
└── requirements.txt
```

## 🎯 Utilisation

### Démarrage de l'Application

```bash
# Démarrer Ollama (si utilisation locale)
ollama serve

# Lancer l'application
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

### Workflow d'Utilisation

1. **Upload de Document**
   - Cliquez sur "Téléchargez un document (PDF)"
   - Sélectionnez votre fichier PDF
   - Cliquez sur "Traiter le document"

2. **Traitement Automatique**
   - Le système extrait et segmente le contenu
   - Crée une base de connaissances vectorielle
   - Initialise le modèle de langue

3. **Posez vos Questions**
   - Utilisez la zone de chat pour poser des questions
   - Le système recherche les informations pertinentes
   - Génère des réponses basées sur le contexte

### Exemples d'Utilisation

**Pour la documentation technique :**
```
"Quelles sont les spécifications système requises ?"
```

**Pour les manuels d'utilisation :**
```
"Comment configurer la connexion réseau ?"
```

**Pour la documentation API :**
```
"Quels sont les paramètres de l'endpoint /users ?"
```