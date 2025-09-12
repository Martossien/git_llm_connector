# 🚀 Git LLM Connector - Tool Open WebUI

Un tool sophistiqué pour Open WebUI qui révolutionne l'interaction avec les dépôts Git en combinant clonage intelligent, analyse par LLM CLI externe, et injection contextuelle automatique.

## 🎯 Vue d'ensemble

Le Git LLM Connector va au-delà des simples connecteurs GitHub existants en offrant :

- **🔄 Clonage et synchronisation** automatique des dépôts GitHub/GitLab
- **🤖 Analyse intelligente** par LLM CLI externes (Qwen, Gemini)
- **📋 Génération de synthèses** structurées (ARCHITECTURE.md, API_SUMMARY.md, CODE_MAP.md)
- **💾 Injection contextuelle** automatique dans les conversations
- **⚡ Performance optimisée** avec cache local et opérations asynchrones

## 🏗️ Architecture

```
~/OW_tools/
├── git_llm_connector.py      # Tool principal
├── git_repos/                # Dépôts clonés
│   └── {owner}_{repo}/
│       ├── .git/             # Dépôt Git
│       ├── docs_analysis/    # Synthèses générées
│       │   ├── ARCHITECTURE.md
│       │   ├── API_SUMMARY.md
│       │   ├── CODE_MAP.md
│       │   └── analysis_metadata.json
│       └── [fichiers du repo]
├── logs/                     # Logs détaillés
│   └── git_llm_connector_YYYYMMDD.log
├── README.md                 # Cette documentation
└── requirements.txt          # Dépendances Python
```

## 🔧 Installation

### Prérequis système

1. **Git** installé et configuré :
```bash
git --version
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

2. **LLM CLI** au choix :

**Option A - Qwen CLI :**
```bash
pip install qwen-cli
# ou suivez les instructions sur https://github.com/QwenLM/Qwen
```

**Option B - Gemini CLI :**
```bash
pip install google-generativeai
# et configurez votre clé API
```

### Installation du Tool

1. **Clonez ou téléchargez** ce dépôt dans votre répertoire Open WebUI tools

2. **Installez les dépendances Python :**
```bash
cd ~/OW_tools
pip install -r requirements.txt
```

3. **Configurez les permissions :**
```bash
chmod +x git_llm_connector.py
```

4. **Ajoutez le tool dans Open WebUI :**
   - Copiez `git_llm_connector.py` dans votre dossier Tools Open WebUI
   - Redémarrez Open WebUI
   - Le tool apparaîtra dans l'interface

## ⚙️ Configuration

### Configuration Administrateur (Valves)

Accessible via l'interface admin d'Open WebUI :

| Paramètre | Défaut | Description |
|-----------|---------|-------------|
| `git_repos_path` | `~/OW_tools/git_repos` | Répertoire de stockage des repos |
| `default_timeout` | `300` | Timeout opérations (secondes) |
| `default_globs_include` | `**/*.py,**/*.js,**/*.ts,...` | Patterns fichiers inclus |
| `default_globs_exclude` | `**/.git/**,**/node_modules/**,...` | Patterns fichiers exclus |
| `max_file_size_kb` | `500` | Taille max fichiers analysés (Ko) |
| `enable_debug_logging` | `true` | Activation logs détaillés |
| `supported_git_hosts` | `github.com,gitlab.com` | Hôtes Git supportés |

### Configuration Utilisateur (UserValves)

Personnalisable par chaque utilisateur :

| Paramètre | Défaut | Description |
|-----------|---------|-------------|
| `llm_cli_choice` | `qwen` | LLM CLI à utiliser (qwen/gemini/auto) |
| `enable_auto_analysis` | `true` | Analyse automatique lors du clone |
| `max_context_files` | `10` | Nb max fichiers injectés |
| `custom_globs_include` | `""` | Patterns personnalisés (inclusion) |
| `custom_globs_exclude` | `""` | Patterns personnalisés (exclusion) |
| `analysis_depth` | `standard` | Profondeur analyse (quick/standard/deep) |
| `preferred_language` | `fr` | Langue des synthèses (fr/en) |

## 🚀 Utilisation

### Fonctions principales

#### 1. `analyze_repo(repo_url)`

Analyse complète d'un dépôt Git :

```python
# Exemples d'URLs supportées
analyze_repo("https://github.com/owner/repo")
analyze_repo("https://gitlab.com/owner/project")
analyze_repo("github.com/owner/repo")  # HTTPS ajouté automatiquement
analyze_repo("git@github.com:owner/repo.git")  # Format SSH
```

**Processus automatique :**
1. 🔍 Parse et validation de l'URL
2. 📥 Clone ou mise à jour du dépôt
3. 📊 Scan des fichiers selon les patterns
4. 🤖 Analyse via LLM CLI (si activée)
5. 📋 Génération des synthèses
6. 💾 Injection dans le contexte

#### 2. `sync_repo(repo_name)`

Synchronisation d'un dépôt existant :

```python
sync_repo("facebook_react")  # Format: owner_repo
```

#### 3. `list_analyzed_repos()`

Liste tous les dépôts analysés avec métadonnées :

```python
list_analyzed_repos()
```

#### 4. `get_repo_context(repo_name)`

Réinjecte le contexte d'un dépôt :

```python
get_repo_context("microsoft_vscode")
```

### Exemples d'usage

**Analyse d'un nouveau projet :**
```
Utilisateur: Peux-tu analyser le dépôt https://github.com/vercel/next.js ?
Assistant: analyze_repo("https://github.com/vercel/next.js")
```

**Questions sur le code après analyse :**
```
Utilisateur: Explique-moi l'architecture de Next.js
Assistant: [Utilise automatiquement le contexte injecté depuis ARCHITECTURE.md]
```

**Synchronisation périodique :**
```
Utilisateur: Met à jour le repo Next.js
Assistant: sync_repo("vercel_next.js")
```

## 📊 Fichiers de synthèse générés

### ARCHITECTURE.md
- Stack technique identifiée
- Modules et composants principaux
- Points d'entrée de l'application
- Organisation du code
- Patterns architecturaux

### API_SUMMARY.md  
- APIs publiques exposées
- Fonctions et méthodes principales
- Interfaces et classes importantes
- Points d'entrée programmatiques

### CODE_MAP.md
- Rôle de chaque dossier principal
- Fichiers critiques à connaître
- Flux de données identifiés
- Guide de navigation dans le code

### analysis_metadata.json
- Informations techniques de l'analyse
- Timestamp et version
- Configuration utilisée
- Statistiques de traitement

## 🔍 Système de logging

### Localisation des logs
```bash
# Logs quotidiens avec rotation
~/OW_tools/logs/git_llm_connector_YYYYMMDD.log
```

### Niveaux de logging
- **DEBUG** : Détails techniques complets
- **INFO** : Étapes principales du processus  
- **WARNING** : Avertissements non bloquants
- **ERROR** : Erreurs avec stack trace

### Exemple de log
```
2024-09-12 14:30:15 - GitLLMConnector - INFO - [analyze_repo:120] - 🚀 Démarrage analyse repo: https://github.com/facebook/react
2024-09-12 14:30:16 - GitLLMConnector - DEBUG - [_parse_git_url:250] - URL parsée: {'owner': 'facebook', 'repo': 'react', 'host': 'github.com'}
2024-09-12 14:30:45 - GitLLMConnector - INFO - [_run_llm_analysis:380] - Analyse LLM terminée: 3 synthèses générées
```

## 🛠️ Dépannage

### Erreurs courantes

#### "Git non disponible"
```bash
# Vérification
git --version

# Installation Ubuntu/Debian
sudo apt-get install git

# Installation CentOS/RHEL  
sudo yum install git
```

#### "LLM CLI non trouvé"
```bash
# Test Qwen CLI
qwen --version

# Test Gemini CLI  
python -c "import google.generativeai; print('OK')"
```

#### "Erreur de permissions"
```bash
# Permissions répertoire
chmod -R 755 ~/OW_tools/
mkdir -p ~/OW_tools/git_repos ~/OW_tools/logs
```

#### "Timeout opérations Git"
Augmentez `default_timeout` dans la configuration admin si vous travaillez avec de gros dépôts.

### Logs de débogage

Pour un débogage approfondi, activez `enable_debug_logging` dans les Valves admin et consultez :

```bash
tail -f ~/OW_tools/logs/git_llm_connector_$(date +%Y%m%d).log
```

## 🔒 Sécurité et bonnes pratiques

### Authentification Git
- Les credentials Git sont gérés par votre configuration locale
- Aucun stockage de mots de passe dans le tool
- Support des clés SSH et tokens personnels GitHub/GitLab

### Isolation des données
- Chaque dépôt est isolé dans son propre dossier
- Les synthèses sont stockées localement uniquement
- Aucune transmission de code vers des APIs externes sans votre contrôle

### Limitations de ressources
- Taille maximale des fichiers configurable
- Patterns d'exclusion pour éviter les fichiers binaires
- Timeouts configurables pour éviter les blocages

## 📈 Performance et optimisation

### Cache intelligent
- Les dépôts clonés sont réutilisés
- Détection automatique des changements
- Analyse incrémentale quand possible

### Opérations asynchrones  
- Clone en arrière-plan avec progression
- Analyses LLM non-bloquantes
- Interface temps réel via event emitters

### Gestion mémoire
- Traitement par chunks des gros fichiers
- Libération automatique des ressources
- Rotation des logs pour éviter l'encombrement

## 🤝 Contribution et développement

### Architecture du code
- **Classes principales** : `Tools`, `Valves`, `UserValves`
- **Méthodes publiques** : 4 fonctions principales exposées
- **Méthodes privées** : 15+ fonctions utilitaires internes
- **Gestion d'erreurs** : Try/catch exhaustif avec logging

### Extension du tool
Pour ajouter support d'autres LLM CLI, modifiez :
1. `_test_llm_cli()` pour la détection
2. `_execute_llm_cli()` pour l'exécution  
3. `test_commands` pour les commandes de test

### Tests
```bash
# Tests manuels recommandés
python -c "
import asyncio
from git_llm_connector import Tools
tool = Tools()
print('Tool initialisé avec succès')
"
```

## 📄 Licence et crédits

- **Licence** : MIT
- **Auteur** : Claude Code Assistant  
- **Version** : 1.0.0
- **Compatibilité** : Open WebUI 0.6.0+

---

## 💡 Questions fréquentes

**Q: Puis-je utiliser d'autres LLM CLI que Qwen/Gemini ?**
R: Actuellement seuls Qwen et Gemini sont supportés. L'architecture permet d'ajouter facilement d'autres LLM CLI en modifiant les méthodes `_test_llm_cli` et `_execute_llm_cli`.

**Q: Les dépôts privés sont-ils supportés ?**
R: Oui, si votre configuration Git locale peut y accéder (clés SSH, tokens). Le tool utilise votre configuration Git existante.

**Q: Quelle est la taille maximale de dépôt supportée ?**
R: Aucune limite stricte, mais les gros dépôts (>1GB) peuvent nécessiter d'ajuster les timeouts et patterns d'exclusion.

**Q: Puis-je personnaliser les prompts d'analyse ?**
R: Actuellement les prompts sont intégrés dans le code. Une future version pourrait permettre la personnalisation via configuration.

**Q: Le tool fonctionne-t-il hors ligne ?**
R: Une fois les dépôts clonés et les LLM CLI installés localement, oui. Seule la synchronisation initiale nécessite une connexion.

---

*Pour un support technique, consultez les logs détaillés dans `~/OW_tools/logs/` et vérifiez votre configuration des LLM CLI.*