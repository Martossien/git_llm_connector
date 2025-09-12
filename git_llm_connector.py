"""
title: Git LLM Connector
author: Martossien
author_url: https://github.com/Martossien
git_url: https://github.com/Martossien/git_llm_connector
description: Tool Open WebUI pour cloner, analyser et résumer des dépôts Git à l'aide de LLM accessibles via les CLI Gemini ou Qwen.
required_open_webui_version: 0.6.0
version: 0.1.0
license: MIT
requirements: aiofiles pathspec pydantic
"""

from typing import Optional, Callable, Awaitable, Any, List, Dict, Union
from pydantic import BaseModel, Field
import asyncio
import aiofiles
import os
import json
import subprocess
import shutil
import glob
import pathspec
import logging
import re
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path

class Tools:
    """
    Git LLM Connector - Tool Open WebUI pour l'analyse intelligente de dépôts Git
    
    Ce tool permet de :
    - Cloner et synchroniser des dépôts GitHub/GitLab
    - Analyser automatiquement le code via des LLM CLI externes
    - Générer des synthèses structurées (ARCHITECTURE.md, API_SUMMARY.md, etc.)
    - Injecter intelligemment le contexte dans les conversations
    
    Architecture :
    ~/OW_tools/
    ├── git_llm_connector.py (ce fichier)
    ├── git_repos/
    │   └── {owner}_{repo}/
    │       ├── docs_analysis/
    │       │   ├── ARCHITECTURE.md
    │       │   ├── API_SUMMARY.md
    │       │   ├── CODE_MAP.md
    │       │   └── analysis_metadata.json
    │       └── [fichiers du repo]
    ├── logs/
    └── README.md
    """
    
    class Valves(BaseModel):
        """
        Configuration globale du Git LLM Connector (niveau administrateur)
        
        Ces paramètres définissent le comportement par défaut du tool
        et peuvent être ajustés par l'administrateur Open WebUI.
        """
        
        git_repos_path: str = Field(
            default="~/OW_tools/git_repos",
            description="Répertoire racine de stockage des dépôts Git clonés"
        )
        
        default_timeout: int = Field(
            default=300,
            description="Timeout par défaut pour les opérations Git et LLM CLI (secondes)"
        )
        
        default_globs_include: str = Field(
            default="**/*.py,**/*.js,**/*.ts,**/*.jsx,**/*.tsx,**/*.vue,**/*.go,**/*.rs,**/*.java,**/*.cpp,**/*.c,**/*.h,**/*.md,**/*.txt,**/*.yml,**/*.yaml,**/*.json,**/*.toml,**/*.cfg,**/*.ini",
            description="Patterns de fichiers à inclure par défaut dans l'analyse"
        )
        
        default_globs_exclude: str = Field(
            default="**/.git/**,**/node_modules/**,**/dist/**,**/build/**,**/__pycache__/**,**/target/**,**/.venv/**,**/venv/**,**/*.png,**/*.jpg,**/*.jpeg,**/*.gif,**/*.svg,**/*.ico,**/*.pdf,**/*.zip,**/*.tar.gz",
            description="Patterns de fichiers à exclure par défaut de l'analyse"
        )
        
        max_file_size_kb: int = Field(
            default=500,
            description="Taille maximale des fichiers à analyser (Ko)"
        )
        
        enable_debug_logging: bool = Field(
            default=True,
            description="Activer les logs détaillés pour le débogage"
        )
        
        supported_git_hosts: str = Field(
            default="github.com,gitlab.com",
            description="Hôtes Git supportés (séparés par des virgules)"
        )

        max_context_bytes: int = Field(
            default=32 * 1024 * 1024,
            description="Taille maximale du contexte envoyé au LLM (octets)"
        )

        max_bytes_per_file: int = Field(
            default=512 * 1024,
            description="Taille maximale lue par fichier pour le contexte (octets)"
        )

        extra_bin_dirs: str = Field(
            default="",
            description="Chemins additionnels pour les binaires LLM (séparés par ':')"
        )

        git_timeout_s: float = Field(
            default=120.0,
            description="Timeout pour les opérations Git (secondes)"
        )

        llm_timeout_s: float = Field(
            default=180.0,
            description="Timeout pour les appels LLM CLI (secondes)"
        )

    class UserValves(BaseModel):
        """
        Configuration utilisateur du Git LLM Connector
        
        Ces paramètres permettent à chaque utilisateur de personnaliser
        le comportement du tool selon ses préférences.
        """
        
        llm_cli_choice: str = Field(
            default="qwen",
            description="Choix du LLM CLI pour l'analyse (qwen, gemini, ou auto)"
        )
        
        enable_auto_analysis: bool = Field(
            default=True,
            description="Activer l'analyse automatique par LLM CLI lors du clonage"
        )
        
        max_context_files: int = Field(
            default=10,
            description="Nombre maximum de fichiers de synthèse à injecter dans le contexte"
        )
        
        custom_globs_include: str = Field(
            default="",
            description="Patterns personnalisés de fichiers à inclure (laissez vide pour utiliser les défauts)"
        )
        
        custom_globs_exclude: str = Field(
            default="",
            description="Patterns personnalisés de fichiers à exclure (laissez vide pour utiliser les défauts)"
        )
        
        analysis_depth: str = Field(
            default="standard",
            description="Profondeur d'analyse (quick, standard, deep)"
        )
        
        preferred_language: str = Field(
            default="fr",
            description="Langue préférée pour les synthèses générées (fr, en)"
        )

        llm_bin_name: str = Field(
            default="gemini",
            description="Nom du binaire LLM à utiliser (gemini ou qwen)"
        )

        llm_model_name: str = Field(
            default="gemini-2.5-pro",
            description="Nom du modèle LLM à utiliser"
        )

        llm_cmd_template: str = Field(
            default="{bin} --model {model} --prompt {prompt}",
            description="Gabarit de commande pour l'appel LLM CLI"
        )

    def __init__(self):
        """
        Initialise le Git LLM Connector
        
        Configure les valves, initialise le système de logging,
        et prépare l'environnement de travail.
        """
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.citation = False  # Open WebUI gère automatiquement les citations
        
        # Initialisation du système de logging
        self._setup_logging()
        
        # Cache des métadonnées des repos analysés
        self._repo_cache = {}
        
        # Patterns de prompts pour les différents LLM CLI
        self._analysis_prompts = {
            "architecture": {
                "fr": "Analyse l'architecture de ce projet de code. Décris la stack technique utilisée, les modules principaux, les points d'entrée de l'application, l'organisation générale du code, et les patterns architecturaux identifiés. Sois concis mais complet.",
                "en": "Analyze the architecture of this code project. Describe the technical stack used, main modules, application entry points, general code organization, and identified architectural patterns. Be concise but comprehensive."
            },
            "api": {
                "fr": "Extrait et documente toutes les APIs, fonctions publiques, classes principales et leurs interfaces dans ce projet. Crée un résumé des points d'entrée programmatiques disponibles.",
                "en": "Extract and document all APIs, public functions, main classes and their interfaces in this project. Create a summary of available programmatic entry points."
            },
            "codemap": {
                "fr": "Crée une carte synthétique du code de ce projet : décris le rôle de chaque dossier principal, identifie les fichiers les plus importants, et explique les flux de données principaux. Aide-moi à naviguer efficacement dans ce codebase.",
                "en": "Create a synthetic code map of this project: describe the role of each main folder, identify the most important files, and explain the main data flows. Help me navigate efficiently through this codebase."
            }
        }
        
        self.logger.info("Git LLM Connector initialisé avec succès")

    def _setup_logging(self) -> None:
        """
        Configure le système de logging avancé
        
        Crée un logger personnalisé avec rotation des fichiers,
        niveaux de verbosité configurables, et formatage structuré.
        """
        # Expansion du répertoire home et création du dossier logs
        log_dir = os.path.expanduser("~/OW_tools/logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Nom du fichier de log avec timestamp
        log_file = os.path.join(
            log_dir, 
            f"git_llm_connector_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        # Configuration du logger principal
        self.logger = logging.getLogger("GitLLMConnector")
        self.logger.setLevel(logging.DEBUG if self.valves.enable_debug_logging else logging.INFO)
        
        # Éviter les handlers dupliqués
        if not self.logger.handlers:
            # Handler pour fichier avec rotation
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Handler pour console (optionnel selon environnement)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # Formatage structuré des logs
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"Système de logging configuré - Fichier: {log_file}")

    async def analyze_repo(
        self, 
        repo_url: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Fonction principale d'analyse d'un dépôt Git
        
        Cette fonction orchestre l'ensemble du processus :
        1. Parse et validation de l'URL
        2. Clone ou mise à jour du dépôt
        3. Scan des fichiers selon les patterns
        4. Analyse via LLM CLI externe
        5. Génération des fichiers de synthèse
        6. Injection dans le contexte
        
        Args:
            repo_url (str): URL complète du dépôt Git (GitHub/GitLab)
            __event_emitter__: Fonction d'émission d'événements Open WebUI
            
        Returns:
            str: Résumé de l'analyse avec statistiques et contexte injecté
            
        Raises:
            ValueError: Si l'URL est invalide ou non supportée
            RuntimeError: Si l'analyse échoue
        """
        try:
            self.logger.info(f"🚀 Démarrage analyse repo: {repo_url}")
            
            # Émission du statut de démarrage
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "🔍 Analyse du dépôt Git en cours...",
                        "done": False,
                        "hidden": False
                    }
                })
            
            # 1. Parse et validation de l'URL
            repo_info = await self._parse_git_url(repo_url)
            self.logger.debug(f"Info repo parsée: {repo_info}")
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"📥 Clonage de {repo_info['owner']}/{repo_info['repo']}...",
                        "done": False,
                        "hidden": False
                    }
                })
            
            # 2. Clone ou mise à jour du dépôt
            local_path = await self._clone_or_update_repo(repo_info, __event_emitter__)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "📊 Scan des fichiers du projet...",
                        "done": False,
                        "hidden": False
                    }
                })
            
            # 3. Scan des fichiers selon les patterns
            file_stats = await self._scan_repository_files(local_path)
            self.logger.info(f"Fichiers scannés: {file_stats['total_files']} fichiers, {file_stats['total_size_mb']:.1f} MB")
            
            # 4. Analyse LLM CLI si activée
            synthesis_files = []
            if self.user_valves.enable_auto_analysis:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"🤖 Analyse par {self.user_valves.llm_cli_choice.upper()}...",
                            "done": False,
                            "hidden": False
                        }
                    })
                
                synthesis_files = await self._run_llm_analysis(local_path, repo_info, __event_emitter__)
            
            # 5. Injection du contexte
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "📋 Injection du contexte...",
                        "done": False,
                        "hidden": False
                    }
                })
            
            context_content = await self._inject_repository_context(local_path, repo_info, synthesis_files, __event_emitter__)
            
            # 6. Finalisation
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "✅ Analyse terminée avec succès !",
                        "done": True,
                        "hidden": False
                    }
                })
            
            # Préparation du résumé final
            summary = f"""
## 📊 Analyse du dépôt {repo_info['owner']}/{repo_info['repo']} terminée

**Statistiques :**
- 📁 Fichiers analysés : {file_stats['total_files']}
- 📦 Taille totale : {file_stats['total_size_mb']:.1f} MB
- 🗂️ Types de fichiers : {', '.join(file_stats['file_types'][:5])}
- 🤖 LLM utilisé : {self.user_valves.llm_cli_choice.upper()}
- 📋 Synthèses générées : {len(synthesis_files)}

Le contexte du dépôt a été injecté et est maintenant disponible pour vos questions !
"""
            
            self.logger.info(f"✅ Analyse terminée avec succès pour {repo_url}")
            return summary
            
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'analyse du dépôt: {str(e)}"
            self.logger.error(f"Erreur analyse repo {repo_url}: {e}", exc_info=True)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": error_msg,
                        "done": True,
                        "hidden": False
                    }
                })
            
            return error_msg

    async def sync_repo(
        self,
        repo_name: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Synchronise un dépôt déjà cloné
        
        Met à jour un dépôt existant et relance l'analyse si des changements
        sont détectés. Optimisé pour les mises à jour incrémentales.
        
        Args:
            repo_name (str): Nom du repo au format "owner_repo"
            __event_emitter__: Fonction d'émission d'événements
            
        Returns:
            str: Résultat de la synchronisation
        """
        try:
            self.logger.info(f"🔄 Synchronisation repo: {repo_name}")
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"🔄 Synchronisation de {repo_name}...",
                        "done": False,
                        "hidden": False
                    }
                })
            
            # Vérification existence du repo local
            repos_path = os.path.expanduser(self.valves.git_repos_path)
            repo_path = os.path.join(repos_path, repo_name)
            
            if not os.path.exists(repo_path):
                raise ValueError(f"Dépôt {repo_name} non trouvé localement")
            
            # Git pull
            result = await self._run_git_command(repo_path, ["pull", "origin"], __event_emitter__)
            
            # Vérification des changements
            has_changes = "Already up to date" not in result
            
            if has_changes and self.user_valves.enable_auto_analysis:
                # Relancer l'analyse
                synthesis_files = await self._run_llm_analysis(
                    repo_path, 
                    {"owner": repo_name.split("_")[0], "repo": "_".join(repo_name.split("_")[1:])},
                    __event_emitter__
                )
                
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "✅ Synchronisation et analyse terminées !",
                            "done": True,
                            "hidden": False
                        }
                    })
                
                return f"✅ Dépôt {repo_name} synchronisé et ré-analysé ({len(synthesis_files)} synthèses mises à jour)"
            else:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "✅ Synchronisation terminée (aucun changement)",
                            "done": True,
                            "hidden": False
                        }
                    })
                
                return f"✅ Dépôt {repo_name} déjà à jour"
                
        except Exception as e:
            error_msg = f"❌ Erreur synchronisation {repo_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": error_msg,
                        "done": True,
                        "hidden": False
                    }
                })
            
            return error_msg

    async def list_analyzed_repos(
        self,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Liste tous les dépôts analysés avec leurs métadonnées
        
        Parcourt le répertoire des dépôts et collecte les informations
        sur chaque analyse effectuée.
        
        Returns:
            str: Liste formatée des dépôts avec statistiques
        """
        try:
            self.logger.info("📋 Listing des dépôts analysés")
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "📋 Collecte des informations des dépôts...",
                        "done": False,
                        "hidden": False
                    }
                })
            
            repos_path = os.path.expanduser(self.valves.git_repos_path)
            
            if not os.path.exists(repos_path):
                return "📁 Aucun dépôt analysé pour le moment"
            
            repos = []
            for item in os.listdir(repos_path):
                repo_path = os.path.join(repos_path, item)
                if os.path.isdir(repo_path):
                    metadata = await self._get_repo_metadata(repo_path)
                    repos.append({
                        "name": item,
                        "path": repo_path,
                        "metadata": metadata
                    })
            
            if not repos:
                return "📁 Aucun dépôt analysé pour le moment"
            
            # Formatage de la liste
            result = f"## 📋 Dépôts Git analysés ({len(repos)} total)\n\n"
            
            for repo in sorted(repos, key=lambda x: x['metadata'].get('last_analysis', '')):
                meta = repo['metadata']
                result += f"### 📦 {repo['name']}\n"
                result += f"- 🕒 Dernière analyse : {meta.get('last_analysis', 'Non disponible')}\n"
                result += f"- 📁 Fichiers : {meta.get('file_count', 'N/A')}\n"
                result += f"- 💾 Taille : {meta.get('total_size_mb', 'N/A')} MB\n"
                result += f"- 🤖 LLM : {meta.get('llm_used', 'N/A')}\n"
                result += f"- 📋 Synthèses : {meta.get('synthesis_count', 0)}\n\n"
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "✅ Liste générée avec succès",
                        "done": True,
                        "hidden": False
                    }
                })
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Erreur listing repos: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return error_msg

    async def get_repo_context(
        self,
        repo_name: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Récupère et injecte le contexte d'un dépôt spécifique
        
        Charge les fichiers de synthèse d'un dépôt déjà analysé
        et les injecte dans le contexte de la conversation.
        
        Args:
            repo_name (str): Nom du repo au format "owner_repo"
            
        Returns:
            str: Contexte injecté ou message d'erreur
        """
        try:
            self.logger.info(f"📋 Récupération contexte pour: {repo_name}")
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"📋 Chargement du contexte de {repo_name}...",
                        "done": False,
                        "hidden": False
                    }
                })
            
            repos_path = os.path.expanduser(self.valves.git_repos_path)
            repo_path = os.path.join(repos_path, repo_name)
            analysis_path = os.path.join(repo_path, "docs_analysis")
            
            if not os.path.exists(analysis_path):
                return f"❌ Aucune analyse trouvée pour {repo_name}. Lancez d'abord analyze_repo()."
            
            # Chargement des fichiers de synthèse
            synthesis_files = []
            for filename in ["ARCHITECTURE.md", "API_SUMMARY.md", "CODE_MAP.md"]:
                file_path = os.path.join(analysis_path, filename)
                if os.path.exists(file_path):
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        synthesis_files.append({
                            "name": filename,
                            "content": content
                        })
            
            # Injection du contexte via citations
            if synthesis_files and __event_emitter__:
                for syn_file in synthesis_files:
                    await __event_emitter__({
                        "type": "citation",
                        "data": {
                            "document": [syn_file["content"]],
                            "metadata": [{"source": f"{repo_name}/{syn_file['name']}", "html": False}],
                            "source": {"name": f"{repo_name} - {syn_file['name']}"}
                        }
                    })
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "✅ Contexte injecté avec succès",
                        "done": True,
                        "hidden": False
                    }
                })
            
            return f"✅ Contexte de {repo_name} chargé ({len(synthesis_files)} fichiers de synthèse injectés)"
            
        except Exception as e:
            error_msg = f"❌ Erreur chargement contexte {repo_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return error_msg

    async def _parse_git_url(self, url: str) -> Dict[str, str]:
        """
        Parse et valide une URL Git
        
        Extrait les informations owner/repo d'une URL GitHub/GitLab
        et valide que l'hôte est supporté.
        
        Args:
            url (str): URL du dépôt Git
            
        Returns:
            Dict contenant owner, repo, host, et url_clean
            
        Raises:
            ValueError: Si l'URL est invalide ou non supportée
        """
        try:
            # Nettoyage de l'URL
            url_clean = url.strip().rstrip('/')
            
            # Support des URLs avec ou sans protocole
            if not url_clean.startswith(('http://', 'https://', 'git@')):
                url_clean = f"https://{url_clean}"
            
            # Parse de l'URL
            if url_clean.startswith('git@'):
                # Format SSH: git@host:owner/repo.git
                match = re.match(r'git@([^:]+):([^/]+)/([^/]+?)(?:\.git)?$', url_clean)
                if not match:
                    raise ValueError("Format SSH invalide")
                host, owner, repo = match.groups()
                url_clean = f"https://{host}/{owner}/{repo}"
            else:
                # Format HTTPS
                parsed = urlparse(url_clean)
                host = parsed.netloc
                path_parts = [p for p in parsed.path.split('/') if p]
                
                if len(path_parts) < 2:
                    raise ValueError("URL doit contenir owner/repo")
                
                owner, repo = path_parts[0], path_parts[1]
                
                # Suppression du .git si présent
                if repo.endswith('.git'):
                    repo = repo[:-4]
            
            # Validation de l'hôte
            supported_hosts = [h.strip() for h in self.valves.supported_git_hosts.split(',')]
            if host not in supported_hosts:
                raise ValueError(f"Hôte {host} non supporté. Hôtes supportés: {', '.join(supported_hosts)}")
            
            result = {
                "owner": owner,
                "repo": repo,
                "host": host,
                "url_clean": url_clean,
                "repo_name": f"{owner}_{repo}"
            }
            
            self.logger.debug(f"URL parsée: {result}")
            return result
            
        except Exception as e:
            raise ValueError(f"URL Git invalide '{url}': {str(e)}")

    async def _clone_or_update_repo(
        self, 
        repo_info: Dict[str, str],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Clone un nouveau dépôt ou met à jour un existant
        
        Args:
            repo_info (Dict): Informations du dépôt (résultat de _parse_git_url)
            __event_emitter__: Fonction d'émission d'événements
            
        Returns:
            str: Chemin local du dépôt
            
        Raises:
            RuntimeError: Si l'opération Git échoue
        """
        try:
            repos_path = os.path.expanduser(self.valves.git_repos_path)
            os.makedirs(repos_path, exist_ok=True)
            
            local_path = os.path.join(repos_path, repo_info["repo_name"])
            
            if os.path.exists(local_path):
                self.logger.info(f"Mise à jour repo existant: {local_path}")
                # Mise à jour
                await self._run_git_command(local_path, ["fetch", "origin"], __event_emitter__)
                await self._run_git_command(local_path, ["reset", "--hard", "origin/HEAD"], __event_emitter__)
            else:
                self.logger.info(f"Clonage nouveau repo: {repo_info['url_clean']} -> {local_path}")
                # Clone
                await self._run_git_command(
                    repos_path, 
                    ["clone", repo_info["url_clean"], repo_info["repo_name"]], 
                    __event_emitter__
                )
            
            return local_path
            
        except Exception as e:
            raise RuntimeError(f"Échec clone/update repo: {str(e)}")

    async def _run_git_command(
        self, 
        cwd: str, 
        cmd: List[str],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Exécute une commande Git de manière asynchrone
        
        Args:
            cwd (str): Répertoire de travail
            cmd (List[str]): Commande Git à exécuter
            __event_emitter__: Fonction d'émission d'événements
            
        Returns:
            str: Sortie de la commande
            
        Raises:
            RuntimeError: Si la commande échoue
        """
        try:
            full_cmd = ["git"] + cmd
            self.logger.debug(f"Exécution commande Git: {' '.join(full_cmd)} dans {cwd}")
            
            process = await asyncio.create_subprocess_exec(
                *full_cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.valves.default_timeout
            )
            
            if process.returncode != 0:
                error_output = stderr.decode('utf-8')
                raise RuntimeError(f"Commande Git échouée: {error_output}")
            
            result = stdout.decode('utf-8')
            self.logger.debug(f"Résultat Git: {result[:200]}...")
            return result
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout commande Git après {self.valves.default_timeout}s")
        except Exception as e:
            raise RuntimeError(f"Erreur exécution Git: {str(e)}")

    async def _scan_repository_files(self, repo_path: str) -> Dict[str, Any]:
        """
        Scan les fichiers du dépôt selon les patterns configurés
        
        Args:
            repo_path (str): Chemin du dépôt local
            
        Returns:
            Dict avec statistiques des fichiers scannés
        """
        try:
            self.logger.info(f"Scan fichiers repo: {repo_path}")
            
            # Détermination des patterns à utiliser
            include_patterns = (
                self.user_valves.custom_globs_include 
                if self.user_valves.custom_globs_include 
                else self.valves.default_globs_include
            ).split(',')
            
            exclude_patterns = (
                self.user_valves.custom_globs_exclude 
                if self.user_valves.custom_globs_exclude 
                else self.valves.default_globs_exclude
            ).split(',')
            
            # Création des specs pathspec
            include_spec = pathspec.PathSpec.from_lines('gitwildmatch', include_patterns)
            exclude_spec = pathspec.PathSpec.from_lines('gitwildmatch', exclude_patterns)
            
            files_found = []
            total_size = 0
            file_types = set()
            
            # Parcours récursif
            for root, dirs, files in os.walk(repo_path):
                # Filtrage des dossiers exclus
                dirs[:] = [d for d in dirs if not exclude_spec.match_file(os.path.join(root, d))]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_path)
                    
                    # Application des patterns
                    if include_spec.match_file(rel_path) and not exclude_spec.match_file(rel_path):
                        try:
                            file_stat = os.stat(file_path)
                            file_size = file_stat.st_size
                            
                            # Vérification taille maximale
                            if file_size <= self.valves.max_file_size_kb * 1024:
                                files_found.append({
                                    "path": rel_path,
                                    "size": file_size,
                                    "ext": os.path.splitext(file)[1].lower()
                                })
                                total_size += file_size
                                
                                # Collecte des types de fichiers
                                ext = os.path.splitext(file)[1].lower()
                                if ext:
                                    file_types.add(ext[1:])  # Sans le point
                                    
                        except OSError:
                            continue  # Fichier inaccessible
            
            stats = {
                "total_files": len(files_found),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_types": sorted(list(file_types)),
                "files": files_found
            }
            
            self.logger.info(f"Scan terminé: {stats['total_files']} fichiers, {stats['total_size_mb']:.1f} MB")
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur scan fichiers: {e}", exc_info=True)
            raise RuntimeError(f"Échec scan fichiers: {str(e)}")

    async def _run_llm_analysis(
        self, 
        repo_path: str, 
        repo_info: Dict[str, str],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> List[str]:
        """
        Lance l'analyse via LLM CLI externe
        
        Prépare les prompts, exécute le LLM CLI, et sauvegarde les résultats
        dans des fichiers de synthèse structurés.
        
        Args:
            repo_path (str): Chemin du dépôt local
            repo_info (Dict): Informations du dépôt
            __event_emitter__: Fonction d'émission d'événements
            
        Returns:
            List des fichiers de synthèse générés
        """
        try:
            self.logger.info(f"Analyse LLM du repo: {repo_path}")
            
            # Création du dossier d'analyse
            analysis_dir = os.path.join(repo_path, "docs_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Vérification de la disponibilité du LLM CLI
            llm_cli = await self._get_available_llm_cli()
            
            if not llm_cli:
                self.logger.warning("Aucun LLM CLI disponible, analyse ignorée")
                return []
            
            # Préparation du contexte de code
            code_context = await self._prepare_code_context(repo_path)
            
            # Génération des synthèses
            synthesis_files = []
            lang = self.user_valves.preferred_language
            
            analyses_to_run = [
                ("ARCHITECTURE.md", "architecture"),
                ("API_SUMMARY.md", "api"),
                ("CODE_MAP.md", "codemap")
            ]
            
            for filename, analysis_type in analyses_to_run:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"🤖 Génération {filename}...",
                            "done": False,
                            "hidden": False
                        }
                    })
                
                try:
                    prompt = self._analysis_prompts[analysis_type][lang]
                    result = await self._execute_llm_cli(llm_cli, prompt, code_context)
                    
                    if result:
                        file_path = os.path.join(analysis_dir, filename)
                        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                            await f.write(result)
                        
                        synthesis_files.append(filename)
                        self.logger.info(f"Synthèse générée: {filename}")
                    
                except Exception as e:
                    self.logger.error(f"Erreur génération {filename}: {e}")
                    continue
            
            # Sauvegarde des métadonnées
            await self._save_analysis_metadata(analysis_dir, repo_info, llm_cli, len(synthesis_files))
            
            self.logger.info(f"Analyse LLM terminée: {len(synthesis_files)} synthèses générées")
            return synthesis_files
            
        except Exception as e:
            self.logger.error(f"Erreur analyse LLM: {e}", exc_info=True)
            return []

    async def _get_available_llm_cli(self) -> Optional[str]:
        """
        Détecte le LLM CLI disponible
        
        Teste la disponibilité des différents LLM CLI selon la configuration
        utilisateur ou détection automatique.
        
        Returns:
            Nom du LLM CLI disponible ou None
        """
        try:
            choice = self.user_valves.llm_cli_choice.lower()
            
            if choice == "auto":
                # Test automatique dans l'ordre de préférence
                for llm in ["qwen", "gemini"]:
                    if await self._test_llm_cli(llm):
                        self.logger.info(f"LLM CLI détecté automatiquement: {llm}")
                        return llm
                return None
            else:
                # Test du choix utilisateur
                if await self._test_llm_cli(choice):
                    return choice
                else:
                    self.logger.warning(f"LLM CLI {choice} non disponible")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Erreur détection LLM CLI: {e}")
            return None

    async def _test_llm_cli(self, llm_name: str) -> bool:
        """
        Test la disponibilité d'un LLM CLI spécifique
        
        Args:
            llm_name (str): Nom du LLM CLI à tester
            
        Returns:
            bool: True si disponible
        """
        try:
            # Commandes de test selon le LLM
            test_commands = {
                "qwen": ["qwen", "--version"],
                "gemini": ["gemini", "--version"]
            }
            
            if llm_name not in test_commands:
                return False
            
            cmd = test_commands[llm_name]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
            
        except Exception:
            return False

    def _redact_secrets(self, text: str) -> str:
        """Masque les secrets communs dans le texte fourni.

        Cette redaction simple remplace les valeurs sensibles par '****'.
        """
        try:
            # Paires clef=valeur classiques (.env)
            text = re.sub(
                r"(?i)(API_KEY|SECRET|TOKEN|PASSWORD|AWS_SECRET_ACCESS_KEY)\s*=\s*[^\s]+",
                lambda m: f"{m.group(1)}=****",
                text,
            )

            # Jetons GitHub personnels
            text = re.sub(r"ghp_[A-Za-z0-9]+", "ghp_****", text)

            # Tokens JWT basiques
            text = re.sub(
                r"eyJ[\w-]+?\.[\w-]+?\.[\w-]+",
                "****",
                text,
            )

            return text
        except Exception:
            # En cas de problème de regex, retourner le texte original
            return text

    async def _prepare_code_context(self, repo_path: str) -> str:
        """
        Prépare le contexte de code pour l'analyse LLM
        
        Sélectionne et formate les fichiers les plus importants
        du dépôt pour alimenter l'analyse LLM.
        
        Args:
            repo_path (str): Chemin du dépôt
            
        Returns:
            str: Contexte formaté pour le LLM
        """
        try:
            max_ctx = self.valves.max_context_bytes
            max_file = self.valves.max_bytes_per_file

            context_parts: List[str] = []
            total_bytes = 0

            marker = "[... CONTEXTE TRONQUÉ ...]"
            marker_bytes = len(marker.encode("utf-8"))

            def append_part(part: str) -> bool:
                nonlocal total_bytes
                part_bytes = len(part.encode("utf-8"))
                if total_bytes + part_bytes > max_ctx:
                    allowed = max_ctx - total_bytes - marker_bytes
                    if allowed > 0:
                        truncated = part.encode("utf-8")[:allowed].decode("utf-8", errors="ignore")
                        context_parts.append(truncated)
                        total_bytes += allowed
                    context_parts.append(marker)
                    total_bytes = max_ctx
                    return False
                context_parts.append(part)
                total_bytes += part_bytes
                return True

            # Fichiers prioritaires (README, package.json, etc.)
            priority_files = [
                "README.md", "README.rst", "README.txt",
                "package.json", "setup.py", "Cargo.toml", "go.mod",
                "requirements.txt", "pyproject.toml", "composer.json"
            ]

            for priority_file in priority_files:
                file_path = os.path.join(repo_path, priority_file)
                if os.path.exists(file_path):
                    try:
                        async with aiofiles.open(file_path, "rb") as f:
                            data = await f.read(max_file)
                        content = data.decode("utf-8", errors="ignore")
                        part = f"=== {priority_file} ===\n{content}\n"
                        if not append_part(part):
                            self.logger.info(f"Contexte tronqué à {total_bytes} octets")
                            return self._redact_secrets("".join(context_parts))
                    except Exception:
                        continue

            # Scan des fichiers du projet selon la profondeur d'analyse
            file_stats = await self._scan_repository_files(repo_path)

            depth_limits = {"quick": 10, "standard": 25, "deep": 50}
            max_files = depth_limits.get(self.user_valves.analysis_depth, 25)
            selected_files = file_stats["files"][:max_files]

            for file_info in selected_files:
                file_path = os.path.join(repo_path, file_info["path"])
                try:
                    async with aiofiles.open(file_path, "rb") as f:
                        data = await f.read(max_file)
                    content = data.decode("utf-8", errors="ignore")
                    part = f"=== {file_info['path']} ===\n{content}\n"
                    if not append_part(part):
                        self.logger.info(f"Contexte tronqué à {total_bytes} octets")
                        return self._redact_secrets("".join(context_parts))
                except Exception:
                    continue

            self.logger.info(f"Contexte total préparé: {total_bytes} octets")
            return self._redact_secrets("".join(context_parts))

        except Exception as e:
            self.logger.error(f"Erreur préparation contexte: {e}")
            return ""

    async def _execute_llm_cli(self, llm_cli: str, prompt: str, context: str) -> Optional[str]:
        """
        Exécute le LLM CLI avec le prompt et le contexte
        
        Args:
            llm_cli (str): Nom du LLM CLI à utiliser
            prompt (str): Prompt d'analyse
            context (str): Contexte de code
            
        Returns:
            Résultat de l'analyse ou None si échec
        """
        try:
            # Préparation de l'input complet
            full_input = f"{prompt}\n\nCONTEXTE DU CODE:\n{context}"
            
            # Commandes selon le LLM CLI
            if llm_cli == "qwen":
                cmd = ["qwen", "--input", "-"]
            elif llm_cli == "gemini":
                cmd = ["gemini", "generate", "--input", "-"]
            else:
                raise ValueError(f"LLM CLI non supporté: {llm_cli}")
            
            # Exécution
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=full_input.encode('utf-8')),
                timeout=self.valves.default_timeout
            )
            
            if process.returncode == 0:
                result = stdout.decode('utf-8').strip()
                self.logger.debug(f"LLM CLI résultat: {len(result)} caractères")
                return result
            else:
                error = stderr.decode('utf-8')
                self.logger.error(f"Erreur LLM CLI: {error}")
                return None
                
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout LLM CLI après {self.valves.default_timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"Erreur exécution LLM CLI: {e}")
            return None

    async def _save_analysis_metadata(
        self, 
        analysis_dir: str, 
        repo_info: Dict[str, str], 
        llm_cli: str, 
        synthesis_count: int
    ) -> None:
        """
        Sauvegarde les métadonnées de l'analyse
        
        Args:
            analysis_dir (str): Répertoire d'analyse
            repo_info (Dict): Informations du dépôt
            llm_cli (str): LLM CLI utilisé
            synthesis_count (int): Nombre de synthèses générées
        """
        try:
            metadata = {
                "repo_info": repo_info,
                "analysis_timestamp": datetime.now().isoformat(),
                "llm_cli_used": llm_cli,
                "synthesis_count": synthesis_count,
                "user_config": {
                    "analysis_depth": self.user_valves.analysis_depth,
                    "preferred_language": self.user_valves.preferred_language
                },
                "tool_version": "1.0.0"
            }
            
            metadata_path = os.path.join(analysis_dir, "analysis_metadata.json")
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
            
            self.logger.info(f"Métadonnées sauvegardées: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde métadonnées: {e}")

    async def _inject_repository_context(
        self, 
        repo_path: str, 
        repo_info: Dict[str, str], 
        synthesis_files: List[str],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Injecte le contexte du dépôt via les citations Open WebUI
        
        Charge les fichiers de synthèse générés et les injecte
        dans la conversation via le système de citations.
        
        Args:
            repo_path (str): Chemin du dépôt
            repo_info (Dict): Informations du dépôt
            synthesis_files (List): Liste des fichiers de synthèse
            __event_emitter__: Fonction d'émission d'événements
            
        Returns:
            str: Résumé du contexte injecté
        """
        try:
            if not __event_emitter__:
                return "Event emitter non disponible pour injection"
            
            analysis_dir = os.path.join(repo_path, "docs_analysis")
            injected_files = 0
            
            # Injection des fichiers de synthèse
            for filename in synthesis_files[:self.user_valves.max_context_files]:
                file_path = os.path.join(analysis_dir, filename)
                
                if os.path.exists(file_path):
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                        
                        # Injection via citation
                        await __event_emitter__({
                            "type": "citation",
                            "data": {
                                "document": [content],
                                "metadata": [{
                                    "source": f"{repo_info['url_clean']}/{filename}",
                                    "html": False
                                }],
                                "source": {
                                    "name": f"{repo_info['owner']}/{repo_info['repo']} - {filename}"
                                }
                            }
                        })
                        
                        injected_files += 1
                        self.logger.debug(f"Contexte injecté: {filename}")
                        
                    except Exception as e:
                        self.logger.error(f"Erreur injection {filename}: {e}")
            
            return f"Contexte injecté: {injected_files} fichiers de synthèse"
            
        except Exception as e:
            self.logger.error(f"Erreur injection contexte: {e}")
            return f"Erreur injection contexte: {str(e)}"

    async def _get_repo_metadata(self, repo_path: str) -> Dict[str, Any]:
        """
        Récupère les métadonnées d'un dépôt analysé
        
        Args:
            repo_path (str): Chemin du dépôt
            
        Returns:
            Dict avec les métadonnées ou valeurs par défaut
        """
        try:
            metadata_path = os.path.join(repo_path, "docs_analysis", "analysis_metadata.json")
            
            if os.path.exists(metadata_path):
                async with aiofiles.open(metadata_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    metadata = json.loads(content)
                    
                    return {
                        "last_analysis": metadata.get("analysis_timestamp", "Inconnue"),
                        "llm_used": metadata.get("llm_cli_used", "Inconnue"),
                        "synthesis_count": metadata.get("synthesis_count", 0),
                        "file_count": "N/A",  # TODO: calculer depuis les stats
                        "total_size_mb": "N/A"
                    }
            else:
                return {
                    "last_analysis": "Métadonnées non disponibles",
                    "llm_used": "Inconnue",
                    "synthesis_count": 0,
                    "file_count": "N/A",
                    "total_size_mb": "N/A"
                }
                
        except Exception as e:
            self.logger.error(f"Erreur lecture métadonnées {repo_path}: {e}")
            return {
                "last_analysis": f"Erreur: {str(e)}",
                "llm_used": "Erreur",
                "synthesis_count": 0,
                "file_count": "N/A",
                "total_size_mb": "N/A"
            }