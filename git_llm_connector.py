"""
title: Git LLM Connector
author: Martossien
author_url: https://github.com/Martossien
git_url: https://github.com/Martossien/git_llm_connector
description: Tool Open WebUI pour cloner, analyser et résumer des dépôts Git à l'aide de LLM accessibles via les CLI Gemini ou Qwen.
required_open_webui_version: 0.6.0
version: 0.2.0
license: MIT
requirements: aiofiles,pathspec,pydantic
"""

TOOL_VERSION = "0.2.0"

from typing import Optional, Callable, Awaitable, Any, List, Dict, Union, Literal, Tuple
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
import shlex
import time
import hashlib

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
        analysis_mode: Literal["smart", "full", "diff"] = Field(
            default="smart",
            description="Mode d'analyse (smart, full, diff)",
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
        focus_paths: str = Field(
            default="",
            description="Sous-chemins relatifs du dépôt à analyser (séparés par des virgules)",
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

        prompt_style: str = Field(
            default="concise, structured, actionable",
            description="Style de rédaction des synthèses",
        )
        prompt_extra_architecture: str = Field(
            default="",
            description="Instructions additionnelles pour l'analyse d'architecture",
        )
        prompt_extra_api: str = Field(
            default="",
            description="Instructions additionnelles pour la synthèse API",
        )
        prompt_extra_codemap: str = Field(
            default="",
            description="Instructions additionnelles pour la carte du code",
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
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        force: bool = False,
        mode: Optional[str] = None,
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
        orig_mode = self.user_valves.analysis_mode
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

            orig_mode = self.user_valves.analysis_mode
            effective_mode = mode if mode else orig_mode
            if force:
                effective_mode = "full"
            self.user_valves.analysis_mode = effective_mode

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
            self.logger.info(
                f"Fichiers scannés: {file_stats['total_files']} fichiers, {file_stats['total_size_mb']:.1f} MB"
            )

            should_reanalyze, prev_meta = await self._should_reanalyze(
                local_path, file_stats["files"], effective_mode, force
            )

            analysis_dir = os.path.join(local_path, "docs_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            synthesis_files: List[str] = []
            llm_info: Dict[str, str] = prev_meta.get("llm", {}) if prev_meta else {}
            synth_count = prev_meta.get("synthesis_count", 0) if prev_meta else 0

            if self.user_valves.enable_auto_analysis and should_reanalyze:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"🤖 Analyse par {self.user_valves.llm_cli_choice.upper()}...",
                            "done": False,
                            "hidden": False,
                        },
                    })
                synthesis_files, llm_info = await self._run_llm_analysis(
                    local_path, repo_info, file_stats, prev_meta, __event_emitter__
                )
                synth_count = len(synthesis_files)
            elif not should_reanalyze:
                self.logger.info(
                    "Aucun changement détecté, réutilisation des synthèses existantes"
                )
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "⚡ Aucun changement détecté, réutilisation des synthèses existantes",
                            "done": False,
                            "hidden": False,
                        },
                    })
                existing = [
                    f
                    for f in ["ARCHITECTURE.md", "API_SUMMARY.md", "CODE_MAP.md"]
                    if os.path.exists(os.path.join(analysis_dir, f))
                ]
                synthesis_files = existing

            await self._save_analysis_metadata(
                local_path,
                repo_info,
                file_stats,
                llm_info,
                synth_count,
                prev_meta if not should_reanalyze else None,
            )

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

            context_content = await self._inject_repository_context(
                local_path, repo_info, synthesis_files, __event_emitter__
            )
            
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
            llm_used = (
                f"{llm_info.get('cli_name', 'N/A').upper()} ({llm_info.get('model', '')})"
                if llm_info
                else "N/A"
            )
            synth_display = f"{len(synthesis_files)} ({'générées' if should_reanalyze else 'réutilisées'})"
            summary = f"""
## 📊 Analyse du dépôt {repo_info['owner']}/{repo_info['repo']} terminée

**Statistiques :**
- 📁 Fichiers analysés : {file_stats['total_files']}
- 📦 Taille totale : {file_stats['total_size_mb']:.1f} MB
- 🗂️ Types de fichiers : {', '.join(file_stats['file_types'][:5])}
- 🤖 LLM utilisé : {llm_used}
- 📋 Synthèses : {synth_display}

Le contexte du dépôt a été injecté et est maintenant disponible pour vos questions !
"""
            
            self.logger.info(f"✅ Analyse terminée avec succès pour {repo_url}")
            return summary
            
        except Exception as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            error_msg = f"❌ Erreur lors de l'analyse du dépôt: {msg}"
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
        finally:
            self.user_valves.analysis_mode = orig_mode

    async def sync_repo(
        self,
        repo_name: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        force: bool = False,
        mode: Optional[str] = None,
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
        orig_mode = self.user_valves.analysis_mode
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
            
            await self._run_git_command(repo_path, ["pull", "origin"], __event_emitter__)

            effective_mode = mode if mode else orig_mode
            if force:
                effective_mode = "full"
            self.user_valves.analysis_mode = effective_mode

            file_stats = await self._scan_repository_files(repo_path)
            should_reanalyze, prev_meta = await self._should_reanalyze(
                repo_path, file_stats["files"], effective_mode, force
            )

            repo_info = {
                "owner": repo_name.split("_")[0],
                "repo": "_".join(repo_name.split("_")[1:]),
            }

            llm_info = prev_meta.get("llm", {}) if prev_meta else {}
            synth_count = prev_meta.get("synthesis_count", 0) if prev_meta else 0
            synthesis_files: List[str] = []

            if self.user_valves.enable_auto_analysis and should_reanalyze:
                synthesis_files, llm_info = await self._run_llm_analysis(
                    repo_path, repo_info, file_stats, prev_meta, __event_emitter__
                )
                synth_count = len(synthesis_files)
                msg = f"✅ Dépôt {repo_name} synchronisé et ré-analysé ({len(synthesis_files)} synthèses mises à jour)"
            else:
                msg = f"✅ Dépôt {repo_name} déjà à jour"

            await self._save_analysis_metadata(
                repo_path,
                repo_info,
                file_stats,
                llm_info,
                synth_count,
                prev_meta if not should_reanalyze else None,
            )

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": msg,
                        "done": True,
                        "hidden": False,
                    },
                })

            return msg

        except Exception as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            error_msg = f"❌ Erreur synchronisation {repo_name}: {msg}"
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
        finally:
            self.user_valves.analysis_mode = orig_mode

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

    async def _git_head(self, repo_path: str) -> str:
        out = await self._run_git_command(repo_path, ["rev-parse", "HEAD"])
        return out.strip()

    def _is_binary_file(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                sample = f.read(2048)
            if b"\0" in sample:
                return True
            try:
                sample.decode("utf-8")
                return False
            except UnicodeDecodeError:
                return True
        except Exception:
            return True

    def _compute_file_sha256(self, abs_path: str, cap_bytes: int) -> str:
        h = hashlib.sha256()
        with open(abs_path, "rb") as f:
            h.update(f.read(cap_bytes))
        return h.hexdigest()

    async def _load_metadata(self, repo_path: str) -> Optional[dict]:
        meta_path = os.path.join(repo_path, "docs_analysis", "analysis_metadata.json")
        if not os.path.exists(meta_path):
            return None
        try:
            async with aiofiles.open(meta_path, "r", encoding="utf-8") as f:
                return json.loads(await f.read())
        except Exception:
            return None

    async def _should_reanalyze(
        self,
        repo_path: str,
        scanned_files: List[Dict[str, Any]],
        mode: str,
        force: bool = False,
    ) -> Tuple[bool, dict]:
        """Détermine si une ré-analyse est nécessaire."""
        if force or mode == "full":
            prev = await self._load_metadata(repo_path)
            return True, prev or {}

        prev = await self._load_metadata(repo_path)
        if not prev:
            return True, {}
        try:
            current_head = await self._git_head(repo_path)
        except Exception:
            return True, prev
        if prev.get("repo_head_commit") and prev["repo_head_commit"] != current_head:
            return True, prev
        if mode in ("smart", "diff"):
            cap = self.valves.max_bytes_per_file
            prev_map = {f["path"]: f.get("sha256") for f in prev.get("files", [])}
            for f in scanned_files:
                cur_sha = f.get("sha256")
                if not cur_sha:
                    abs_p = os.path.join(repo_path, f["path"])
                    try:
                        cur_sha = self._compute_file_sha256(abs_p, cap)
                    except Exception:
                        return True, prev
                if prev_map.get(f["path"]) != cur_sha:
                    return True, prev
            return False, prev
        return True, prev

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

            files_found: List[Dict[str, Any]] = []
            total_size = 0
            file_types = set()

            focus_paths = [p.strip().strip('/\\') for p in self.user_valves.focus_paths.split(',') if p.strip()]
            roots = [repo_path] if not focus_paths else []
            for fp in focus_paths:
                abs_fp = os.path.join(repo_path, fp)
                if os.path.isdir(abs_fp):
                    roots.append(abs_fp)
            if not roots:
                roots = [repo_path]

            cap = self.valves.max_bytes_per_file

            for base in roots:
                for root, dirs, files in os.walk(base):
                    dirs[:] = [
                        d
                        for d in dirs
                        if not exclude_spec.match_file(os.path.relpath(os.path.join(root, d), repo_path))
                    ]

                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, repo_path)

                        if include_spec.match_file(rel_path) and not exclude_spec.match_file(rel_path):
                            try:
                                file_stat = os.stat(file_path)
                                file_size = file_stat.st_size
                                if file_size <= self.valves.max_file_size_kb * 1024:
                                    try:
                                        sha = self._compute_file_sha256(file_path, cap)
                                    except Exception:
                                        sha = ""
                                    files_found.append({
                                        "path": rel_path,
                                        "size": file_size,
                                        "ext": os.path.splitext(file)[1].lower(),
                                        "sha256": sha,
                                    })
                                    total_size += file_size
                                    ext = os.path.splitext(file)[1].lower()
                                    if ext:
                                        file_types.add(ext[1:])
                            except OSError:
                                continue
            
            stats = {
                "total_files": len(files_found),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_types": sorted(list(file_types)),
                "files": files_found,
                "include_patterns": include_patterns,
                "exclude_patterns": exclude_patterns,
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
        file_stats: Dict[str, Any],
        prev_metadata: Optional[dict],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Exécute l'analyse LLM et retourne les fichiers générés et info LLM."""
        try:
            self.logger.info(f"Analyse LLM du repo: {repo_path}")

            analysis_dir = os.path.join(repo_path, "docs_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            llm_cli = await self._get_available_llm_cli()
            if not llm_cli:
                self.logger.warning("Aucun LLM CLI disponible, analyse ignorée")
                return [], {}

            bin_name = (
                self.user_valves.llm_bin_name
                if llm_cli == self.user_valves.llm_cli_choice.lower()
                else llm_cli
            )
            bin_path = self._resolve_executable(bin_name)
            if not bin_path:
                msg = f"Binaire LLM introuvable: {bin_name}"
                self.logger.error(msg)
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": msg[:200], "done": True, "hidden": False},
                    })
                return [], {}
            self.logger.info(f"Binaire LLM utilisé: {bin_path}")

            code_context, changed_count = await self._prepare_code_context(
                repo_path, file_stats, prev_metadata
            )
            if (
                self.user_valves.analysis_mode == "diff"
                and changed_count == 0
                and __event_emitter__
            ):
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Aucun fichier modifié détecté, contexte limité aux fichiers prioritaires",
                        "done": False,
                        "hidden": False,
                    },
                })

            synthesis_files = []
            lang = self.user_valves.preferred_language
            analyses_to_run = [
                ("ARCHITECTURE.md", "architecture"),
                ("API_SUMMARY.md", "api"),
                ("CODE_MAP.md", "codemap"),
            ]

            for filename, analysis_type in analyses_to_run:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"🤖 Génération {filename}...",
                            "done": False,
                            "hidden": False,
                        },
                    })
                try:
                    base_prompt = self._analysis_prompts[analysis_type].get(
                        lang, self._analysis_prompts[analysis_type]["en"]
                    )
                    extra = getattr(
                        self.user_valves, f"prompt_extra_{analysis_type}", ""
                    )
                    prompt = (
                        f"{base_prompt}\n\nStyle: {self.user_valves.prompt_style}\n"
                        f"Extra: {extra}\nLanguage: {lang}"
                    )
                    result = await self._execute_llm_cli(
                        llm_cli, bin_path, prompt, code_context, __event_emitter__
                    )
                    if result:
                        file_path = os.path.join(analysis_dir, filename)
                        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                            await f.write(result)
                        synthesis_files.append(filename)
                        self.logger.info(f"Synthèse générée: {filename}")
                except Exception as e:
                    self.logger.error(f"Erreur génération {filename}: {e}")
                    continue

            llm_info = {
                "cli_name": llm_cli,
                "bin_path": bin_path,
                "model": self.user_valves.llm_model_name,
            }
            self.logger.info(
                f"Analyse LLM terminée: {len(synthesis_files)} synthèses générées"
            )
            return synthesis_files, llm_info
        except Exception as e:
            self.logger.error(f"Erreur analyse LLM: {e}", exc_info=True)
            return [], {}

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

    def _resolve_executable(self, name: str) -> Optional[str]:
        """Résout le chemin absolu d'un exécutable.

        Recherche dans le PATH courant puis dans les répertoires
        additionnels définis par ``extra_bin_dirs`` (séparés par ``:``).

        Args:
            name: Nom de l'exécutable à rechercher.

        Returns:
            Chemin absolu de l'exécutable ou ``None`` si introuvable.
        """
        path = shutil.which(name)
        if path:
            return path

        extra = self.valves.extra_bin_dirs
        if extra:
            for d in extra.split(":"):
                d = d.strip()
                if not d:
                    continue
                candidate = shutil.which(os.path.join(os.path.expanduser(d), name))
                if candidate:
                    return candidate
        return None

    async def _test_llm_cli(self, llm_name: str) -> bool:
        """Test la disponibilité d'un LLM CLI spécifique."""
        try:
            # Utilise un binaire personnalisé si l'utilisateur a choisi ce LLM
            bin_name = (
                self.user_valves.llm_bin_name
                if llm_name == self.user_valves.llm_cli_choice.lower()
                else llm_name
            )
            bin_path = self._resolve_executable(bin_name)
            if not bin_path:
                return False

            process = await asyncio.create_subprocess_exec(
                bin_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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

    async def _prepare_code_context(
        self,
        repo_path: str,
        file_stats: Optional[Dict[str, Any]] = None,
        prev_metadata: Optional[dict] = None,
    ) -> Tuple[str, int]:
        """
        Prépare le contexte de code pour l'analyse LLM.

        Returns:
            Tuple[str, int]: Contexte formaté et nombre de fichiers ajoutés.
        """
        try:
            max_ctx = self.valves.max_context_bytes
            max_file = self.valves.max_bytes_per_file

            context_parts: List[str] = []
            total_bytes = 0
            changed_files = 0

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

            priority_files = [
                "README.md", "README.rst", "README.txt",
                "package.json", "setup.py", "Cargo.toml", "go.mod",
                "requirements.txt", "pyproject.toml", "composer.json"
            ]

            for priority_file in priority_files:
                file_path = os.path.join(repo_path, priority_file)
                if os.path.exists(file_path) and not self._is_binary_file(file_path):
                    try:
                        async with aiofiles.open(file_path, "rb") as f:
                            data = await f.read(max_file)
                        content = data.decode("utf-8", errors="ignore")
                        part = f"=== {priority_file} ===\n{content}\n"
                        if not append_part(part):
                            self.logger.info(f"Contexte tronqué à {total_bytes} octets")
                            return self._redact_secrets("".join(context_parts)), changed_files
                    except Exception:
                        continue

            if file_stats is None:
                file_stats = await self._scan_repository_files(repo_path)

            depth_limits = {"quick": 10, "standard": 25, "deep": 50}
            max_files = depth_limits.get(self.user_valves.analysis_depth, 25)

            files_list = file_stats["files"]
            if (
                self.user_valves.analysis_mode == "diff" and prev_metadata is not None
            ):
                prev_map = {f["path"]: f.get("sha256") for f in prev_metadata.get("files", [])}
                changed = []
                for f in files_list:
                    if prev_map.get(f["path"]) != f.get("sha256"):
                        changed.append(f)
                files_list = changed

            selected_files = files_list[:max_files]

            sem = asyncio.Semaphore(8)

            async def read_file(info: Dict[str, Any]) -> Tuple[str, Optional[str]]:
                path = os.path.join(repo_path, info["path"])
                if self._is_binary_file(path):
                    return info["path"], None
                async with sem:
                    try:
                        async with aiofiles.open(path, "rb") as f:
                            data = await f.read(max_file)
                        return info["path"], data.decode("utf-8", errors="ignore")
                    except Exception:
                        return info["path"], None

            results = await asyncio.gather(*(read_file(f) for f in selected_files))

            for rel_path, content in results:
                if content is None:
                    continue
                part = f"=== {rel_path} ===\n{content}\n"
                if not append_part(part):
                    self.logger.info(f"Contexte tronqué à {total_bytes} octets")
                    return self._redact_secrets("".join(context_parts)), changed_files
                changed_files += 1

            self.logger.info(f"Contexte total préparé: {total_bytes} octets")
            return self._redact_secrets("".join(context_parts)), changed_files

        except Exception as e:
            self.logger.error(f"Erreur préparation contexte: {e}")
            return "", 0

    async def _execute_llm_cli(
        self,
        llm_cli: str,
        bin_path: str,
        prompt: str,
        context: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Optional[str]:
        """Exécute le LLM CLI avec le prompt et le contexte via stdin."""

        if llm_cli == "gemini":
            argv = [
                bin_path,
                "--model",
                self.user_valves.llm_model_name,
                "-p",
                prompt,
            ]
        else:
            argv = [
                bin_path,
                "--model",
                self.user_valves.llm_model_name,
                "--prompt",
                prompt,
            ]

        log_cmd = " ".join(argv[:-1] + ["<prompt>"])
        context_size = len(context.encode("utf-8"))
        self.logger.info(
            f"Exécution LLM: {log_cmd} | contexte {context_size} octets"
        )

        start = time.perf_counter()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=context.encode("utf-8")),
                timeout=self.valves.llm_timeout_s,
            )
            duration = time.perf_counter() - start
            out_text = stdout.decode("utf-8", errors="ignore")
            err_text = stderr.decode("utf-8", errors="ignore")

            if process.returncode != 0:
                self.logger.error(
                    f"LLM CLI échec (code {process.returncode}) après {duration:.1f}s: {err_text.strip()}"
                )
                if __event_emitter__:
                    truncated = (
                        err_text.strip()[:200] + "..."
                        if len(err_text.strip()) > 200
                        else err_text.strip()
                    )
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"Erreur LLM CLI: {truncated}",
                            "done": True,
                            "hidden": False,
                        },
                    })
                return None

            self.logger.info(
                f"LLM CLI terminé en {duration:.1f}s (code {process.returncode})"
            )
            return out_text.strip()

        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            msg = f"Timeout LLM CLI après {self.valves.llm_timeout_s}s"
            self.logger.error(msg)
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": msg, "done": True, "hidden": False},
                })
            return None
        except Exception as e:
            self.logger.error(f"Erreur exécution LLM CLI: {e}", exc_info=True)
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Erreur LLM CLI: {e}", "done": True, "hidden": False},
                })
            return None

    async def _save_analysis_metadata(
        self,
        repo_path: str,
        repo_info: Dict[str, str],
        file_stats: Dict[str, Any],
        llm_info: Dict[str, str],
        synthesis_count: int,
        prev_metadata: Optional[dict] = None,
    ) -> None:
        """Sauvegarde les métadonnées de l'analyse."""
        try:
            analysis_dir = os.path.join(repo_path, "docs_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            try:
                head = await self._git_head(repo_path)
            except Exception:
                head = ""

            if prev_metadata:
                files_meta = prev_metadata.get("files", [])
            else:
                files_meta = [
                    {
                        "path": f["path"],
                        "size": f.get("size", 0),
                        "sha256": f.get("sha256", ""),
                        "analyzed_at": datetime.now().isoformat(),
                    }
                    for f in file_stats.get("files", [])
                ]

            metadata = {
                "repo_info": repo_info,
                "analysis_timestamp": datetime.now().isoformat(),
                "tool_version": TOOL_VERSION,
                "repo_head_commit": head,
                "scan_config": {
                    "include": file_stats.get("include_patterns", []),
                    "exclude": file_stats.get("exclude_patterns", []),
                    "max_file_size_kb": self.valves.max_file_size_kb,
                },
                "files": files_meta,
                "llm": llm_info,
                "synthesis_count": synthesis_count,
            }

            metadata_path = os.path.join(analysis_dir, "analysis_metadata.json")
            async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
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
                    llm_info = metadata.get("llm", {})
                    llm_used = (
                        f"{llm_info.get('cli_name', 'Inconnue')} ({llm_info.get('model', '')})"
                        if llm_info
                        else "Inconnue"
                    )
                    files = metadata.get("files", [])
                    total_size = sum(f.get("size", 0) for f in files)
                    return {
                        "last_analysis": metadata.get("analysis_timestamp", "Inconnue"),
                        "llm_used": llm_used,
                        "synthesis_count": metadata.get("synthesis_count", 0),
                        "file_count": len(files),
                        "total_size_mb": round(total_size / (1024 * 1024), 1),
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
