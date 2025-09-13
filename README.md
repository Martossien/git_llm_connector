# 🚀 Git LLM Connector v1.6 - Tool Open WebUI (Expérimental)

Un outil expérimental pour Open WebUI qui combine clonage Git, analyse de code par LLM CLI locaux, et exploration interactive des dépôts.

## 🎯 Vue d'ensemble

Le Git LLM Connector offre **15 fonctions publiques** pour :

- **📥 Clonage et synchronisation** Git (GitHub/GitLab) en local  
- **🤖 Analyse intelligente** via LLM CLI externes (Qwen, Gemini)
- **📋 Génération de synthèses** structurées (ARCHITECTURE.md, API_SUMMARY.md, CODE_MAP.md)
- **🔍 Exploration interactive** du code (scan, recherche, aperçu)
- **📊 Statistiques** et métadonnées détaillées

### ✨ **Avantages clés du local**
- **🔒 Confidentialité** : votre code reste sur votre machine
- **🚀 Performance** : pas de limite de taux d'API externes
- **💰 Économique** : Qwen CLI gratuit, Gemini CLI généreux (1000 req/jour)
- **📚 Contexte complet** : analysez des projets entiers sans contrainte de tokens
- **🛠️ Contrôle total** : choisissez vos modèles LLM et paramètres

## 🏗️ Architecture réelle

```
~/git_llm_connector/           # Base du système (créé automatiquement)
├── git_repos/                 # Dépôts clonés localement
│   └── {nom_repo}/
│       ├── .git/              # Dépôt Git complet
│       ├── docs_analysis/     # Synthèses LLM générées
│       │   ├── ARCHITECTURE.md
│       │   ├── API_SUMMARY.md
│       │   ├── CODE_MAP.md
│       │   └── analysis_metadata.json
│       └── [fichiers du repo]
└── logs/                      # Logs quotidiens avec rotation
    └── git_llm_connector_YYYYMMDD.log
```

## 🔧 Installation complète

### 1. Prérequis système

**Node.js 20+ requis pour les LLM CLI :**
```bash
# Vérifier si installé
node -v
npm -v

# Installation si nécessaire (Ubuntu/Debian)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Ou avec nvm (recommandé)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install 22
```

**Git installé et configuré :**
```bash
git --version
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

### 2. Installation LLM CLI

#### **Qwen CLI (GRATUIT - recommandé)**
```bash
# Installation
npm install -g @qwen-code/qwen-code

# Vérification
qwen --version

# Premier lancement pour authentification
qwen
# → Se connecter avec compte qwen.ai (gratuit)
# → 2000 requêtes/jour, 60 req/minute
```

#### **Gemini CLI (gratuit avec limites)**
```bash
# Installation  
npm install -g @google/gemini-cli

# Vérification
gemini --version

# Premier lancement pour authentification
gemini
# → Choisir "Login with Google"
# → 1000 requêtes/jour, 60 req/minute
```

⚠️ **Important** : Vous devez vous authentifier avec les CLI avant utilisation !

### 3. Installation dans Open WebUI

1. **Ouvrir Open WebUI** → Aller dans **Workspace** → **Tools**
2. **Cliquer sur le +** pour ajouter un nouveau tool
3. **Effacer le code de test** présent par défaut
4. **Copier tout le contenu** de `git_llm_connector.py` dans l'éditeur
5. **Cliquer "Valider"** pour sauvegarder
6. Le système créera automatiquement `~/git_llm_connector/` au premier lancement

### 4. Activation dans les conversations

1. **Nouvelle conversation** → **Cliquer sur le +** en haut
2. **Activer "Git LLM Connector"** dans la liste des tools
3. **Taper les commandes** directement dans la discussion

## 🚀 Guide d'utilisation détaillé

### 📋 1. Fonctions de diagnostic

#### `tool_health()` - État du système
```python
tool_health()
# → "OK | base_dir=~/git_llm_connector | repos_dir=~/git_llm_connector/git_repos"
```
**🎯 Utilité dev :** Vérifier que l'outil fonctionne correctement après installation.

#### `llm_check(llm, model)` - Test des LLM CLI
```python
llm_check()  # Auto-détection
llm_check("gemini", "gemini-2.5-pro")
# → "✅ LLM détecté: gemini | diag: /usr/bin/gemini --version -> rc=0"
```
**🎯 Utilité dev :** Diagnostiquer les problèmes d'installation ou d'authentification des LLM CLI.

### 📂 2. Gestion des dépôts

#### `git_clone(repo_url, name)` - Clonage local
```python
# Clonage standard
git_clone("https://github.com/microsoft/vscode")
# → Clone dans "microsoft_vscode"

# Nom personnalisé
git_clone("https://github.com/vercel/next.js", "nextjs_main")
# → Clone dans "nextjs_main"

# Formats supportés
git_clone("git@github.com:facebook/react.git", "react_fb")
git_clone("https://gitlab.com/owner/project", "my_project")
```
**🎯 Utilité dev :** Récupérer rapidement n'importe quel projet open source pour étude ou contribution.

#### `git_update(repo_name, strategy)` - Synchronisation
```python
# Mise à jour douce (pull)
git_update("microsoft_vscode")

# Reset dur (écrase les changements locaux)
git_update("nextjs_main", "reset")
```
**🎯 Utilité dev :** Maintenir vos clones à jour sans re-télécharger.

#### `list_repos()` - Inventaire
```python
list_repos()
# → Liste formatée de tous vos dépôts clonés
```
**🎯 Utilité dev :** Vue d'ensemble rapide de tous les projets locaux.

### 🤖 3. Analyse par LLM (⏱️ 5-30 minutes selon taille)

#### `analyze_repo()` - Génération de synthèses IA 
```python
# Analyse complète par défaut
analyze_repo("microsoft_vscode")
# → Génère ARCHITECTURE.md, API_SUMMARY.md, CODE_MAP.md

# Analyse personnalisée complète
analyze_repo(
    repo_name="nextjs_main",
    sections="architecture,api,codemap",  # Ou "architecture" seul
    depth="deep",                         # quick/standard/deep
    language="fr",                        # fr/en  
    llm="gemini",                        # auto/qwen/gemini
    model="gemini-2.5-pro"              # Modèle spécifique
)

# Analyse rapide architecture seulement
analyze_repo("react_fb", sections="architecture", depth="quick", language="en", llm="qwen")

# Focus API pour bibliothèque
analyze_repo("lodash_clone", sections="api", depth="deep")
```
**🎯 Utilité dev :** Comprendre rapidement l'architecture d'un projet complexe avant de contribuer ou s'en inspirer.

⚠️ **Temps d'exécution** : 5-10 min pour petits projets, 15-30 min pour gros projets (React, VSCode, etc.)

#### `clean_analysis(repo_name)` - Nettoyage avant re-analyse
```python
clean_analysis("microsoft_vscode")
```
**🎯 Utilité dev :** Forcer une nouvelle analyse avec des paramètres différents.

### 📊 4. Exploration et métadonnées

#### `repo_info(repo_name)` - Fiche technique
```python
repo_info("microsoft_vscode")
# → Métadonnées, dernière analyse, taille des synthèses
```
**🎯 Utilité dev :** Vérifier qu'une analyse est terminée et voir sa qualité.

#### `list_analyzed_repos()` - Inventaire analysé
```python
list_analyzed_repos()
# → Repos avec synthèses + timestamp + LLM utilisé
```
**🎯 Utilité dev :** Voir d'un coup d'œil quels projets ont été analysés par IA.

#### `get_repo_context(repo_name, max_files, max_chars_per_file)` - Injection dans conversation
```python
# Contexte standard pour discussion
get_repo_context("nextjs_main")

# Contexte léger
get_repo_context("react_fb", max_files=2, max_chars_per_file=1000)
```
**🎯 Utilité dev :** Alimenter la conversation avec les synthèses pour poser des questions intelligentes.

### 🔍 5. Exploration interactive du code

#### `scan_repo_files()` - Vue d'ensemble des fichiers
```python
# Scan standard (par taille, plus gros en premier)
scan_repo_files("microsoft_vscode")

# Top 20 par nom alphabétique
scan_repo_files("nextjs_main", limit=20, order="path", ascending=True)

# Plus petits fichiers d'abord (configs, utils)
scan_repo_files("my_project", limit=30, order="size", ascending=True)
```
**🎯 Utilité dev :** Identifier rapidement les gros fichiers ou naviguer alphabétiquement.

#### `preview_file()` - Aperçu de code
```python
# Voir un fichier de config
preview_file("nextjs_main", "package.json")

# Code source avec limite  
preview_file("react_fb", "src/index.js", max_bytes=2048)

# Fichier dans structure complexe
preview_file("vscode_clone", "src/vs/editor/browser/widget/codeEditorWidget.ts")
```
**🎯 Utilité dev :** Examiner des fichiers clés sans quitter la conversation.

#### `stats_repo()` - Statistiques techniques
```python
# Stats complètes
stats_repo("microsoft_vscode")

# Focus sur top 15 plus gros
stats_repo("nextjs_main", top_n=15)
```
**🎯 Utilité dev :** Comprendre la répartition technique (langages, gros fichiers).

#### `find_in_repo()` - Recherche dans le code
```python
# Recherche simple
find_in_repo("nextjs_main", "useState")

# Regex avancée pour patterns
find_in_repo("react_fb", r"function\s+\w+Component", use_regex=True)

# Recherche de TODOs
find_in_repo("my_project", "TODO", max_matches=20)

# Trouver toutes les routes API
find_in_repo("backend_api", r"@app\.(get|post|put|delete)", use_regex=True, max_matches=50)
```
**🎯 Utilité dev :** Localiser rapidement des patterns, APIs, ou problèmes dans un codebase inconnu.

## 💡 Workflows recommandés pour développeurs

### 🔄 **Découverte d'un nouveau projet**
```python
# 1. Cloner le projet
git_clone("https://github.com/microsoft/playwright")

# 2. Vue d'ensemble rapide  
stats_repo("microsoft_playwright")
scan_repo_files("microsoft_playwright", limit=15)

# 3. Analyse IA complète (☕ pause café - 15 min)
analyze_repo("microsoft_playwright", depth="deep", language="fr")

# 4. Étudier les synthèses
get_repo_context("microsoft_playwright")
# → Maintenant vous pouvez poser des questions intelligentes !
```

### 🔍 **Investigation de bug/feature**
```python
# 1. Chercher le code lié au problème
find_in_repo("my_project", "authentication", max_matches=30)

# 2. Prévisualiser les fichiers suspects
preview_file("my_project", "src/auth/middleware.ts")

# 3. Rechercher des patterns problématiques
find_in_repo("my_project", r"console\.(log|error)", use_regex=True)

# 4. Analyser l'API concernée
analyze_repo("my_project", sections="api", depth="standard")
```

### 🔄 **Mise à jour et re-analyse**
```python  
# 1. Mettre à jour le code
git_update("popular_framework")

# 2. Nettoyer l'ancienne analyse
clean_analysis("popular_framework")

# 3. Re-analyser avec focus
analyze_repo("popular_framework", sections="architecture,codemap", depth="standard")
```

### 🏗️ **Étude architecturale approfondie**
```python
# 1. Architecture générale
analyze_repo("complex_system", sections="architecture", depth="deep")

# 2. Cartographie détaillée  
analyze_repo("complex_system", sections="codemap", depth="deep")

# 3. Exploration des gros composants
stats_repo("complex_system", top_n=20)
find_in_repo("complex_system", r"class\s+\w+", use_regex=True, max_matches=100)
```

## ⚙️ Configuration

### Valves administrateur
- `git_repos_path`: `/home/user/git_llm_connector/git_repos` (fixe)
- `llm_timeout_s`: `900.0` secondes (15 min pour analyses LLM)
- `git_timeout_s`: `180.0` secondes  
- `max_file_size_kb`: `500` Ko par fichier analysé
- `max_context_bytes`: `32 MB` de contexte total

### UserValves (personnalisables)
- `llm_cli_choice`: `qwen` | `gemini` | `auto` (défaut: qwen car gratuit)
- `analysis_depth`: `quick` | `standard` | `deep`  
- `preferred_language`: `fr` | `en`
- `llm_model_name`: modèle par défaut
- `max_context_files`: nombre de synthèses injectées dans conversations

## 📄 Synthèses IA générées

### 📐 ARCHITECTURE.md
- Stack technique et frameworks utilisés
- Organisation des modules principaux  
- Points d'entrée de l'application
- Patterns architecturaux identifiés
- Dépendances critiques

**Exemple d'utilisation :** "Comment Next.js gère-t-il le SSR ?" → Réponse basée sur l'analyse architecturale.

### 🔧 API_SUMMARY.md
- Fonctions et classes exportées publiquement
- Interfaces et types principaux
- Points d'entrée programmatiques
- Documentation des signatures importantes

**Exemple d'utilisation :** "Quelles sont les méthodes principales de React ?" → Liste extraite automatiquement.

### 🗺️ CODE_MAP.md
- Navigation guidée dans la structure
- Rôle de chaque dossier principal
- Fichiers critiques à examiner
- Flux de données principaux
- Guide pour les nouveaux contributeurs

**Exemple d'utilisation :** "Où trouver la logique de routage dans Express ?" → Guidance précise vers les bons fichiers.

## 🛠️ Dépannage

### Tests de diagnostic rapides
```python
tool_health()                    # Vérifier les chemins système
llm_check()                     # Tester détection LLM CLI  
list_repos()                    # Inventaire des dépôts
debug_status("test connexion")  # Test basique
```

### Logs détaillés
```bash
# Voir les logs en temps réel
tail -f ~/git_llm_connector/logs/git_llm_connector_$(date +%Y%m%d).log

# Logs spécifiques d'une date
cat ~/git_llm_connector/logs/git_llm_connector_20250912.log
```

### Erreurs courantes

#### "Git non disponible"
```bash
# Diagnostic
git --version

# Installation Ubuntu/Debian
sudo apt-get update && sudo apt-get install git

# Configuration obligatoire
git config --global user.name "Votre Nom"
git config --global user.email "votre@email.com"
```

#### "LLM CLI non trouvé" 
```bash
# Test Qwen
qwen --version
# Si échec : npm install -g @qwen-code/qwen-code

# Test Gemini  
gemini --version
# Si échec : npm install -g @google/gemini-cli

# Test Node.js
node -v  # Doit être >= 20
```

#### "Timeout analyse LLM"
- Augmenter `llm_timeout_s` dans les Valves admin
- Utiliser `depth="quick"` pour gros projets
- Essayer un autre LLM CLI (`llm="gemini"` si Qwen bloque)

#### "Dépôt inexistant"
```python
list_repos()  # Vérifier le nom exact
repo_info("nom_exact_du_repo")  # Diagnostic complet
```

#### "Authentification LLM échouée"
```bash
# Re-authentification Qwen
qwen  # Suivre le processus de login

# Re-authentification Gemini
gemini  # Choisir "Login with Google"
```

## 📊 Performance et limites

### **Temps d'exécution typiques**
- **Clone Git** : 30s à 5min (selon taille et connexion)
- **Analyse LLM** : 
  - Quick : 2-5 minutes
  - Standard : 5-15 minutes  
  - Deep : 15-30 minutes (projets >100k LOC)
- **Exploration** : instantané (scan, recherche, aperçu)

### **Limites techniques**
- **Fichiers max** : 500 Ko par fichier (configurable)
- **Contexte max** : 32 MB total (configurable)
- **Taux API gratuits** :
  - Qwen : 2000 req/jour, 60/minute ✨
  - Gemini : 1000 req/jour, 60/minute

### **Optimisations recommandées**
- Utilisez `depth="quick"` pour premiers aperçus
- Analysez par sections (`sections="architecture"`) pour aller plus vite
- Nettoyez les anciennes analyses avant re-analyse
- Utilisez les patterns d'exclusion pour ignorer node_modules, etc.

## 🔒 Sécurité et confidentialité

### ✅ **Avantages sécurité**
- **Code reste local** : jamais transmis vers des serveurs inconnus
- **Credentials Git locaux** : utilise votre config SSH/tokens existante
- **Logs anonymisés** : secrets automatiquement masqués
- **Isolation utilisateur** : chaque utilisateur a son espace

### ⚠️ **Points d'attention**
- Les synthèses LLM sont basées sur votre code → gardez les résultats privés si nécessaire
- Authentification LLM CLI requise → utilisez comptes personnels/professionnels appropriés
- Timeouts configurables → évitent les blocages mais limitent les gros projets

## 💰 Coûts et quotas

### **Qwen CLI (recommandé)**
- ✅ **Gratuit** : 2000 requêtes/jour
- ✅ **Pas de limite de tokens** sur les requêtes
- ✅ **Modèles performants** : Qwen3-Coder optimisé pour le code

### **Gemini CLI** 
- ✅ **Gratuit généreux** : 1000 requêtes/jour  
- ✅ **Modèle premium** : Gemini 2.5 Pro (1M contexte)
- ⚠️ **API payante disponible** pour besoins enterprise

### **Comparaison outils payants**
- Claude Code : ~20€/mois
- GitHub Copilot : ~10€/mois  
- **Git LLM Connector** : Gratuit (après setup initial)

---

## 📝 Notes importantes

### ⚙️ **Installation Open WebUI**
1. **Workspace** → **Tools** → **+** 
2. Supprimer code test par défaut
3. Coller le contenu de `git_llm_connector.py`
4. Valider et activer dans conversations avec **+**

### ⏱️ **Patience requise**
- Premier clone + analyse : **20-45 minutes** total
- Analyses suivantes : **5-15 minutes** (code déjà local)
- Exploration interactive : **instantané**

### 🆓 **Pourquoi Qwen par défaut ?**
- **Gratuit** sans limite de tokens
- **Optimisé pour code** (Qwen3-Coder)
- **Performant** sur analyses architecturales
- **2000 requêtes/jour** = ~30 analyses complètes

### 🎯 **Public cible**
- Développeurs explorant de nouveaux projets
- Équipes évaluant des frameworks/libraries  
- Mainteneurs analysant des contributions
- Étudiants apprenant sur du code real-world
- DevOps cartographiant des systèmes existants

---

## 🆕 Fonctions v1.9 (Developer UX)

### `outline_file(repo_name, path, max_items=200)`
**Navigation intelligente de gros fichiers**
```python
outline_file("my_project", "src/large_file.py", 100)
# → Structure avec fonctions, classes, méthodes (ligne + type + nom)
```

### `find_tests_for(repo_name, target, max_results=50)`
**Découverte de tests pour symbole ou fichier**
```python
find_tests_for("my_project", "authenticate")  # Pour une fonction
find_tests_for("my_project", "src/auth.py")   # Pour un fichier
# → Candidats triés par pertinence + extraits de code
```

### `recent_changes(repo_name, days=7, max_commits=50)`
**Résumé git pour contexte de debug**
```python
recent_changes("my_project", 7, 20)
# → Commits récents + fichiers les plus modifiés
```

---

**💡 Conseil :** Commencez par cloner un petit projet familier, faites une analyse `depth="quick"`, et explorez les fonctionnalités avant de vous attaquer aux gros dépôts !

*Pour support : consultez les logs détaillés dans `~/git_llm_connector/logs/` et vérifiez votre config LLM CLI avec `llm_check()`. Voir aussi `docs/PLAYBOOKS_v1.9.md` pour les workflows optimaux.*
