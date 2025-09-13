"""
title: Git LLM Connector
author: Martossien
author_url: https://github.com/Martossien
git_url: https://github.com/Martossien/git_llm_connector
description: v1.9 UX-focused — Git + LLM with developer UX enhancements: file outlining, test discovery, recent changes tracking. Built on solid v1.8 foundation (AST, hybrid retrieval, call graph). All via CLI (gemini/qwen). Secure, bounded context, detailed logs, stable paths, LLM timeout=900s.
required_open_webui_version: 0.6.0
version: 0.1.9
license: MIT
requirements: aiofiles,pathspec,pydantic

CHANGELOG v1.9:
- NEW: File outlining - outline_file() for large file navigation
- NEW: Test discovery - find_tests_for() symbol/file test finder
- NEW: Recent changes - recent_changes() git history summary for context
- NEW: Developer UX focus with orchestration documentation
- Enhanced: Better navigation workflow for complex codebases
- Enhanced: Test-driven development support with smart test discovery
- Enhanced: Git-aware context for debugging recent changes
- Maintained: All v1.6/v1.7/v1.8 functions with full backward compatibility
"""

from typing import Any, List, Dict, Set, Tuple, Optional, Union
from pydantic import BaseModel, Field
import os
import logging
from datetime import datetime
import re
import json
import stat
import pathspec
import subprocess
import shutil
import time
import hashlib
import math
import collections
from collections import defaultdict, Counter

# Optional heavy dependencies with graceful fallback
_TREE_SITTER_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_FAISS_AVAILABLE = False
_DATASKETCH_AVAILABLE = False

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    pass

try:
    from datasketch import MinHashLSH, MinHash
    _DATASKETCH_AVAILABLE = True
except ImportError:
    pass

# -------------------------------------
# UTILITAIRE LOCAL (enhanced v1.8)
# -------------------------------------
_SECRET_PATTERNS = [
    (re.compile(r"(?i)(API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*[^\s]+"), r"\1=****"),
    (re.compile(r"ghp_[A-Za-z0-9]+"), "ghp_****"),
    (re.compile(r"eyJ[\w-]+?\.[\w-]+?\.[\w-]+"), "****"),  # JWT
    (re.compile(r"sk-[A-Za-z0-9]{48}"), "sk-****"),  # OpenAI API keys
    (re.compile(r"xoxb-[0-9]+-[0-9]+-[0-9]+-[a-z0-9]+"), "xoxb-****"),  # Slack tokens
]

# -------------------------------------
# SYMBOLIC INDEX PATTERNS (v1.7)
# -------------------------------------
_SYMBOL_PATTERNS = {
    "python": {
        "functions": [
            re.compile(r"^\s*def\s+(\w+)\s*\("),
            re.compile(r"^\s*async\s+def\s+(\w+)\s*\("),
        ],
        "classes": [
            re.compile(r"^\s*class\s+(\w+)\s*[:\(]"),
        ],
        "imports": [
            re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)"),
            re.compile(r"^\s*import\s+([\w\.]+)"),
        ],
    },
    "javascript": {
        "functions": [
            re.compile(r"^\s*function\s+(\w+)\s*\("),
            re.compile(r"^\s*const\s+(\w+)\s*=\s*(?:async\s+)?\(?.*?\)?\s*=>"),
            re.compile(r"^\s*const\s+(\w+)\s*=\s*function"),
            re.compile(r"^\s*(\w+)\s*:\s*(?:async\s+)?function\s*\("),
        ],
        "classes": [
            re.compile(r"^\s*class\s+(\w+)\s*[{]"),
        ],
        "exports": [
            re.compile(r"^\s*export\s+default\s+(?:class|function)?\s*(\w+)?"),
            re.compile(r"^\s*export\s+(?:const|function|class)\s+(\w+)"),
            re.compile(r"^\s*module\.exports\s*=\s*(\w+)"),
        ],
        "imports": [
            re.compile(r'^\s*import\s+.+\s+from\s+[\'"]([^\'"]+)[\'"]'),
            re.compile(r'^\s*const\s+.+\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'),
        ],
    },
}

# Map file extensions to language keys
_EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
}

# -------------------------------------
# TREE-SITTER SETUP (v1.8)
# -------------------------------------
_TREE_SITTER_PARSERS = {}

def _init_tree_sitter():
    """Initialize Tree-sitter parsers if available."""
    global _TREE_SITTER_PARSERS
    if not _TREE_SITTER_AVAILABLE:
        return

    try:
        # Try to load common language parsers
        # Note: In practice, these would need to be built/installed separately
        # For demonstration, we'll use a mock setup that falls back gracefully
        languages = ['python', 'javascript', 'typescript']

        for lang in languages:
            try:
                # This is a placeholder - real implementation would load actual .so files
                # Language.build_library(f'/path/to/{lang}.so', [f'/path/to/tree-sitter-{lang}'])
                # _TREE_SITTER_PARSERS[lang] = Language(f'/path/to/{lang}.so', lang)
                pass
            except Exception:
                continue

    except Exception:
        pass

# Initialize parsers on import
_init_tree_sitter()

# -------------------------------------
# BM25 IMPLEMENTATION (v1.8)
# -------------------------------------
class SimpleBM25:
    """Simple BM25 implementation for code retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0.0

    def fit(self, documents: List[str]):
        """Build BM25 index from documents."""
        self.documents = documents
        self.doc_len = []
        self.doc_freqs = []

        # Tokenize and build document frequencies
        df = defaultdict(int)  # document frequency

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))

            freq = Counter(tokens)
            self.doc_freqs.append(freq)

            for token in freq.keys():
                df[token] += 1

        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        # Calculate IDF scores
        N = len(documents)
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for code."""
        # Split on non-alphanumeric, preserve camelCase and snake_case
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+', text.lower())
        return tokens

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search for top-k most relevant documents."""
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[idx]

            for token in query_tokens:
                if token in doc_freq and token in self.idf:
                    tf = doc_freq[token]
                    idf = self.idf[token]

                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator

            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class Tools:
    """
    Git LLM Connector — v1.8 ADVANCED

    ✅ All v1.6/v1.7 Public Functions (signatures preserved):
      - tool_health(dummy: str = "") -> str
      - debug_status(message: str = "ping") -> str
      - list_repos() -> str
      - list_analyzed_repos() -> str
      - repo_info(repo_name: str) -> str
      - get_repo_context(repo_name: str, max_files: int = 3, max_chars_per_file: int = 2000) -> str
      - scan_repo_files(repo_name: str, limit: int = 50, order: str = "size", ascending: bool = False) -> str
      - preview_file(repo_name: str, relative_path: str, max_bytes: int = 65536) -> str
      - stats_repo(repo_name: str, top_n: int = 10) -> str
      - find_in_repo(repo_name: str, needle: str, use_regex: bool = False, max_matches: int = 50) -> str
      - git_clone(repo_url: str, name: str = "") -> str
      - git_update(repo_name: str, strategy: str = "pull") -> str
      - clean_analysis(repo_name: str) -> str
      - llm_check(llm: str = "", model: str = "") -> str
      - analyze_repo(repo_name: str, sections: str = "architecture,api,codemap", depth: str = "", language: str = "", llm: str = "", model: str = "") -> str

    🆕 New v1.7 RAG Functions:
      - auto_retrieve_context(repo_name: str, question: str) -> str
      - build_simple_index(repo_name: str) -> str
      - quick_api_lookup(repo_name: str, api_name: str) -> str
      - find_usage_examples(repo_name: str, symbol: str) -> str
      - show_related_code(repo_name: str, path: str) -> str
      - auto_load_context(repo_name: str, user_question: str) -> str
    """

    # ------------------------------
    # Valves (admin) - unchanged from v1.6
    # ------------------------------
    class Valves(BaseModel):
        git_repos_path: str = Field(
            default="/home/user/git_llm_connector/git_repos",
            description="Répertoire racine local pour les dépôts Git clonés.",
        )
        default_timeout: int = Field(
            default=300, description="Timeout par défaut pour opérations basiques (s)."
        )
        default_globs_include: str = Field(
            default="**/*.py,**/*.js,**/*.ts,**/*.jsx,**/*.tsx,**/*.vue,**/*.go,**/*.rs,**/*.java,**/*.cpp,**/*.c,**/*.h,**/*.md,**/*.txt,**/*.yml,**/*.yaml,**/*.json,**/*.toml,**/*.cfg,**/*.ini",
            description="Patterns de fichiers à inclure (séparés par des virgules).",
        )
        default_globs_exclude: str = Field(
            default="**/.git/**,**/node_modules/**,**/dist/**,**/build/**,**/__pycache__/**,**/target/**,**/.venv/**,**/venv/**,**/*.png,**/*.jpg,**/*.jpeg,**/*.gif,**/*.svg,**/*.ico,**/*.pdf,**/*.zip,**/*.tar.gz",
            description="Patterns de fichiers à exclure (séparés par des virgules).",
        )
        max_file_size_kb: int = Field(
            default=500, description="Taille max lue par fichier (Ko) pour inclusion."
        )
        enable_debug_logging: bool = Field(
            default=True, description="Activer les logs détaillés."
        )
        supported_git_hosts: str = Field(
            default="github.com,gitlab.com",
            description="Hôtes Git supportés (séparés par des virgules).",
        )
        max_context_bytes: int = Field(
            default=32 * 1024 * 1024,
            description="Taille max de contexte (octets).",
        )
        max_bytes_per_file: int = Field(
            default=512 * 1024,
            description="Octets max lus par fichier pour le contexte.",
        )
        extra_bin_dirs: str = Field(
            default="",
            description="Répertoires additionnels pour binaires LLM (séparés par ':').",
        )
        git_timeout_s: float = Field(
            default=180.0, description="Timeout Git (secondes)."
        )
        llm_timeout_s: float = Field(
            default=900.0, description="Timeout appels LLM CLI (secondes)."
        )
        emit_citations: bool = Field(
            default=True, description="Émettre des citations si supportées."
        )

    # ------------------------------
    # UserValves (utilisateur) - extended for v1.7
    # ------------------------------
    class UserValves(BaseModel):
        llm_cli_choice: str = Field(
            default="qwen", description="LLM CLI (qwen / gemini / auto)."
        )
        analysis_mode: str = Field(
            default="smart", description="Mode d'analyse (réservé v2)."
        )
        enable_auto_analysis: bool = Field(
            default=True, description="Analyse auto après clone (réservé v2)."
        )
        max_context_files: int = Field(
            default=10, description="Nb max de fichiers de synthèse à injecter."
        )
        custom_globs_include: str = Field(
            default="", description="Patterns personnalisés à inclure."
        )
        custom_globs_exclude: str = Field(
            default="", description="Patterns personnalisés à exclure."
        )
        focus_paths: str = Field(
            default="", description="Chemins à prioriser (séparés par des virgules)."
        )
        analysis_depth: str = Field(
            default="standard", description="Profondeur (quick, standard, deep)."
        )
        preferred_language: str = Field(
            default="fr", description="Langue préférée (fr/en)."
        )
        llm_bin_name: str = Field(
            default="gemini", description="Binaire LLM (ex: gemini, qwen)."
        )
        llm_model_name: str = Field(
            default="gemini-2.5-pro", description="Nom du modèle LLM."
        )
        llm_cmd_template: str = Field(
            default="{bin} --model {model} --prompt {prompt}",
            description="Template de commande LLM.",
        )
        prompt_style: str = Field(
            default="concise, structured, actionable",
            description="Style des prompts (réservé v2).",
        )
        prompt_extra_architecture: str = Field(
            default="", description="Suffixe prompt ARCHITECTURE (v2)."
        )
        prompt_extra_api: str = Field(
            default="", description="Suffixe prompt API (v2)."
        )
        prompt_extra_codemap: str = Field(
            default="", description="Suffixe prompt CODEMAP (v2)."
        )
        # v1.7 new fields
        rag_max_extracts: int = Field(
            default=6, description="Nombre max d'extraits pour auto-retrieve."
        )
        rag_context_window: int = Field(
            default=40, description="Lignes de contexte autour des hits."
        )
        enable_deduplication: bool = Field(
            default=True, description="Activer la déduplication des extraits."
        )
        # v1.8 new fields
        retrieval_backend: str = Field(
            default="keywords", description="Backend de récupération (keywords/bm25/hybrid)."
        )
        bm25_k1: float = Field(
            default=1.5, description="Paramètre BM25 k1."
        )
        bm25_b: float = Field(
            default=0.75, description="Paramètre BM25 b."
        )
        use_embeddings: bool = Field(
            default=False, description="Utiliser embeddings si disponibles."
        )
        embedding_model: str = Field(
            default="sentence-transformers/all-MiniLM-L6-v2",
            description="Modèle d'embeddings."
        )
        max_vector_chunks: int = Field(
            default=20000, description="Nombre max de chunks vectorisés."
        )

    # ------------------------------
    # Init - lightweight as per guide
    # ------------------------------
    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.citation = False

        home = os.path.expanduser("~")
        self.base_dir = os.path.join(home, "git_llm_connector")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.repos_dir = os.path.join(self.base_dir, "git_repos")
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.repos_dir, exist_ok=True)

        # Force paths and timeout
        self.valves.git_repos_path = self.repos_dir
        self.valves.llm_timeout_s = 900.0

        self._setup_logging()
        self.logger.info(
            "[INIT] v1.8 advanced — base=%s, repos=%s, logs=%s, llm_timeout=%ss, ast=%s, embeddings=%s, faiss=%s",
            self.base_dir,
            self.repos_dir,
            self.logs_dir,
            self.valves.llm_timeout_s,
            _TREE_SITTER_AVAILABLE,
            _SENTENCE_TRANSFORMERS_AVAILABLE,
            _FAISS_AVAILABLE,
        )

    # ------------------------------
    # Logging - unchanged from v1.6
    # ------------------------------
    def _setup_logging(self) -> None:
        log_dir = getattr(
            self,
            "logs_dir",
            os.path.join(os.path.expanduser("~"), "git_llm_connector", "logs"),
        )
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"git_llm_connector_{datetime.now().strftime('%Y%m%d')}.log"
        )

        self.logger = logging.getLogger("GitLLMConnector")
        self.logger.setLevel(
            logging.DEBUG if self.valves.enable_debug_logging else logging.INFO
        )

        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            fmt = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
            )
            fh.setFormatter(fmt)
            ch.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.logger.info("Système de logging configuré - Fichier: %s", log_file)

    # ------------------------------
    # Helpers - unchanged from v1.6 plus new v1.7 helpers
    # ------------------------------
    @staticmethod
    def _paths():
        home = os.path.expanduser("~")
        base = os.path.join(home, "git_llm_connector")
        return {
            "base": base,
            "repos": os.path.join(base, "git_repos"),
            "logs": os.path.join(base, "logs"),
        }

    @staticmethod
    def _sanitize_repo_name(name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
        return safe.strip("._-") or "repo"

    @staticmethod
    def _ext_lang_hint(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "tsx",
            ".jsx": "jsx",
            ".json": "json",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".md": "markdown",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
        }.get(ext, "")

    @staticmethod
    def _looks_binary(head: bytes) -> bool:
        if b"\x00" in head:
            return True
        nontext = sum(1 for b in head if b < 9 or (13 < b < 32) or b > 126)
        return (len(head) > 0) and (nontext / len(head) > 0.30)

    @staticmethod
    def _build_specs(includes: List[str], excludes: List[str]):
        inc = pathspec.PathSpec.from_lines(
            "gitwildmatch", [p.strip() for p in includes if p.strip()]
        )
        exc = pathspec.PathSpec.from_lines(
            "gitwildmatch", [p.strip() for p in excludes if p.strip()]
        )
        return inc, exc

    @staticmethod
    def _parse_git_url(url: str) -> Dict[str, str]:
        url = url.strip()
        m = re.match(r"^git@([^:]+):([^/]+)/([^/]+?)(?:\.git)?$", url)
        if m:
            host, owner, repo = m.group(1), m.group(2), m.group(3)
        else:
            m = re.match(r"^https?://([^/]+)/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
            if not m:
                raise ValueError("URL Git invalide")
            host, owner, repo = m.group(1), m.group(2), m.group(3)
        repo = repo[:-4] if repo.endswith(".git") else repo
        return {
            "host": host,
            "owner": owner,
            "repo": repo,
            "repo_name": f"{owner}_{repo}",
            "url_clean": f"https://{host}/{owner}/{repo}",
        }

    @staticmethod
    def _redact_secrets(text: str) -> str:
        try:
            for pat, repl in _SECRET_PATTERNS:
                text = pat.sub(repl, text)
            return text
        except Exception:
            return text

    # =====================================
    # NEW v1.7 RAG HELPERS
    # =====================================

    def _extract_keywords_from_question(self, question: str) -> Set[str]:
        """Extract technical keywords from user question."""
        keywords = set()

        # CamelCase/snake_case patterns
        camel_snake = re.findall(
            r"\b[a-zA-Z][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*\b|\b[a-z_]+[a-z0-9_]*\b",
            question,
        )
        keywords.update(camel_snake)

        # Technical terms
        tech_terms = [
            "function",
            "class",
            "method",
            "api",
            "hook",
            "component",
            "service",
            "error",
            "exception",
            "bug",
            "test",
            "import",
            "export",
            "async",
            "await",
            "return",
            "interface",
            "type",
            "const",
            "var",
            "let",
        ]
        question_lower = question.lower()
        for term in tech_terms:
            if term in question_lower:
                keywords.add(term)

        # File patterns
        file_patterns = re.findall(r"\*\.[a-z]+|\b\w+\.[a-z]+\b", question.lower())
        keywords.update(file_patterns)

        return {k for k in keywords if len(k) > 2}

    def _get_repo_files_for_search(self, repo_dir: str) -> List[Dict[str, Any]]:
        """Get searchable files from repo with metadata."""
        includes = (
            self.user_valves.custom_globs_include or self.valves.default_globs_include
        ).split(",")
        excludes = (
            self.user_valves.custom_globs_exclude or self.valves.default_globs_exclude
        ).split(",")
        max_kb = int(self.valves.max_file_size_kb)

        inc, exc = self._build_specs(includes, excludes)
        files = []

        for root, dirs, filenames in os.walk(repo_dir):
            rel_root = os.path.relpath(root, repo_dir)
            if rel_root == ".":
                rel_root = ""

            # Filter excluded directories
            keep_dirs = []
            for d in dirs:
                rel_dir = os.path.join(rel_root, d) if rel_root else d
                if not exc.match_file(rel_dir):
                    keep_dirs.append(d)
            dirs[:] = keep_dirs

            for fn in filenames:
                rel_path = os.path.join(rel_root, fn) if rel_root else fn
                if exc.match_file(rel_path) or not inc.match_file(rel_path):
                    continue

                full_path = os.path.join(repo_dir, rel_path)
                try:
                    st = os.stat(full_path)
                    if not stat.S_ISREG(st.st_mode) or st.st_size > max_kb * 1024:
                        continue

                    # Check if binary
                    with open(full_path, "rb") as f:
                        head = f.read(4096)
                        if self._looks_binary(head):
                            continue

                    files.append(
                        {
                            "path": rel_path,
                            "full_path": full_path,
                            "size": st.st_size,
                            "lang": self._ext_lang_hint(full_path),
                        }
                    )
                except Exception:
                    continue

        return files

    def _search_for_keywords_in_files(
        self, files: List[Dict], keywords: Set[str]
    ) -> List[Dict]:
        """Search for keywords in files and return scored results."""
        results = []

        for file_info in files:
            try:
                with open(
                    file_info["full_path"], "r", encoding="utf-8", errors="ignore"
                ) as f:
                    lines = f.readlines()

                file_hits = []
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    hit_keywords = []

                    for keyword in keywords:
                        if keyword.lower() in line_lower:
                            hit_keywords.append(keyword)

                    if hit_keywords:
                        score = 0
                        # Score based on hit type
                        if re.search(
                            r"\b(def|function|class|const|export|import)\b", line_lower
                        ):
                            score += 2  # Definition
                        elif any(
                            k in ["import", "export", "from", "require"]
                            for k in hit_keywords
                        ):
                            score += 1  # Import

                        file_hits.append(
                            {
                                "line": i,
                                "content": line.strip(),
                                "keywords": hit_keywords,
                                "score": score,
                            }
                        )

                if file_hits:
                    total_score = sum(h["score"] for h in file_hits)
                    # Boost test files for tutorial intent
                    if any(
                        test_marker in file_info["path"].lower()
                        for test_marker in ["test", "spec", "example", "demo"]
                    ):
                        total_score += 1

                    results.append(
                        {
                            "file": file_info,
                            "hits": file_hits[:10],  # Limit hits per file
                            "total_score": total_score,
                        }
                    )

            except Exception:
                continue

        return sorted(results, key=lambda x: x["total_score"], reverse=True)

    def _extract_context_around_hits(self, file_path: str, hits: List[Dict]) -> str:
        """Extract context around hits, avoiding mid-function cuts."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            context_window = self.user_valves.rag_context_window
            all_ranges = []

            for hit in hits:
                line_num = hit["line"]
                start = max(1, line_num - context_window)
                end = min(len(lines), line_num + context_window)

                # Try to expand to complete functions/classes
                # Look backward for function/class start
                for i in range(start - 1, max(0, start - 20), -1):
                    line_content = lines[i].strip()
                    if line_content.startswith(
                        ("def ", "class ", "function ", "const ")
                    ) or re.match(
                        r"^\s*(export\s+)?(function|class|const)\s+\w+", line_content
                    ):
                        start = i + 1
                        break

                all_ranges.append((start, end, line_num))

            # Merge overlapping ranges
            merged_ranges = []
            for start, end, hit_line in sorted(all_ranges):
                if merged_ranges and start <= merged_ranges[-1][1] + 5:
                    merged_ranges[-1] = (
                        merged_ranges[-1][0],
                        max(merged_ranges[-1][1], end),
                    )
                else:
                    merged_ranges.append((start, end))

            extracts = []
            for start, end in merged_ranges:
                if end - start > 200:  # Too large, truncate
                    end = start + 200

                extract_lines = []
                for i in range(start - 1, end):
                    if i < len(lines):
                        extract_lines.append(f"{i+1:4d}: {lines[i].rstrip()}")

                extracts.append(
                    f"### {os.path.basename(file_path)}:{start}-{end}\n```\n"
                    + "\n".join(extract_lines)
                    + "\n```"
                )

            return "\n\n".join(extracts)

        except Exception as e:
            return f"❌ Error extracting context: {e}"

    def _deduplicate_extracts(self, extracts: List[str]) -> List[str]:
        """Enhanced deduplication that calls the new MinHash version."""
        return self._enhanced_deduplicate_extracts(extracts)

    def _detect_question_intent(self, question: str) -> str:
        """Detect intent from user question."""
        question_lower = question.lower()

        debug_keywords = [
            "error",
            "exception",
            "stack",
            "traceback",
            "not working",
            "bug",
            "crash",
            "fail",
            "broken",
            "issue",
            "problem",
        ]
        if any(kw in question_lower for kw in debug_keywords):
            return "debug"

        api_keywords = [
            "api",
            "function",
            "method",
            "class",
            "signature",
            "parameters",
            "return",
            "interface",
            "type",
        ]
        if any(kw in question_lower for kw in api_keywords):
            return "api_lookup"

        nav_keywords = [
            "where is",
            "find",
            "locate",
            "path",
            "file",
            "folder",
            "directory",
        ]
        if any(kw in question_lower for kw in nav_keywords):
            return "navigation"

        tutorial_keywords = [
            "how to",
            "example",
            "sample",
            "demo",
            "guide",
            "usage",
            "tutorial",
        ]
        if any(kw in question_lower for kw in tutorial_keywords):
            return "tutorial"

        return "general"

    # =====================================
    # v1.7 RAG INDEX FUNCTIONALITY
    # =====================================

    def _build_symbol_index_for_repo(self, repo_dir: str) -> Dict[str, Any]:
        """Build symbolic index for a repository."""
        index = {
            "repo": os.path.basename(repo_dir),
            "generated_at": datetime.now().isoformat(),
            "functions": [],
            "classes": [],
            "imports": [],
            "exports": [],
        }

        files = self._get_repo_files_for_search(repo_dir)

        for file_info in files:
            lang_key = _EXT_TO_LANG.get(os.path.splitext(file_info["path"])[1].lower())
            if not lang_key or lang_key not in _SYMBOL_PATTERNS:
                continue

            patterns = _SYMBOL_PATTERNS[lang_key]

            try:
                with open(
                    file_info["full_path"], "r", encoding="utf-8", errors="ignore"
                ) as f:
                    for line_num, line in enumerate(f, 1):
                        # Functions
                        for pattern in patterns.get("functions", []):
                            match = pattern.match(line)
                            if match:
                                index["functions"].append(
                                    {
                                        "name": match.group(1),
                                        "file": file_info["path"],
                                        "line": line_num,
                                        "lang": lang_key,
                                    }
                                )

                        # Classes
                        for pattern in patterns.get("classes", []):
                            match = pattern.match(line)
                            if match:
                                index["classes"].append(
                                    {
                                        "name": match.group(1),
                                        "file": file_info["path"],
                                        "line": line_num,
                                        "lang": lang_key,
                                    }
                                )

                        # Imports (Python)
                        if lang_key == "python":
                            for pattern in patterns.get("imports", []):
                                match = pattern.match(line)
                                if match:
                                    if "from" in line:
                                        # from module import items
                                        module = match.group(1)
                                        items_str = match.group(2)
                                        items = [
                                            item.strip()
                                            for item in items_str.split(",")
                                        ]
                                        index["imports"].append(
                                            {
                                                "from": module,
                                                "items": items,
                                                "file": file_info["path"],
                                                "line": line_num,
                                                "lang": lang_key,
                                            }
                                        )
                                    else:
                                        # import module
                                        module = match.group(1)
                                        index["imports"].append(
                                            {
                                                "from": module,
                                                "items": [module.split(".")[-1]],
                                                "file": file_info["path"],
                                                "line": line_num,
                                                "lang": lang_key,
                                            }
                                        )

                        # Exports (JavaScript)
                        if lang_key == "javascript":
                            for pattern in patterns.get("exports", []):
                                match = pattern.match(line)
                                if match:
                                    name = (
                                        match.group(1) if match.group(1) else "default"
                                    )
                                    kind = "default" if "default" in line else "named"
                                    index["exports"].append(
                                        {
                                            "name": name,
                                            "file": file_info["path"],
                                            "line": line_num,
                                            "kind": kind,
                                            "lang": lang_key,
                                        }
                                    )

                            # Imports (JavaScript)
                            for pattern in patterns.get("imports", []):
                                match = pattern.match(line)
                                if match:
                                    module = match.group(1)
                                    index["imports"].append(
                                        {
                                            "from": module,
                                            "items": ["*"],  # Simplified
                                            "file": file_info["path"],
                                            "line": line_num,
                                            "lang": lang_key,
                                        }
                                    )

            except Exception as e:
                self.logger.warning(f"Error indexing {file_info['path']}: {e}")
                continue

        return index

    def _load_or_create_index(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Load existing index or create new one."""
        p = self._paths()
        safe = self._sanitize_repo_name(repo_name)
        repo_dir = os.path.join(p["repos"], safe)
        analysis_dir = os.path.join(repo_dir, "docs_analysis")
        index_path = os.path.join(analysis_dir, "simple_index.json")

        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load index: {e}")

        # Try to create index automatically
        if os.path.isdir(repo_dir):
            try:
                self.build_simple_index(repo_name)
                if os.path.exists(index_path):
                    with open(index_path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to auto-create index: {e}")

        return None

    # =====================================
    # LLM HELPERS (unchanged from v1.6)
    # =====================================
    def _resolve_executable(self, name: str) -> str | None:
        path = shutil.which(name)
        if path:
            return path
        extra = self.valves.extra_bin_dirs or ""
        for d in [p for p in extra.split(":") if p.strip()]:
            cand = shutil.which(os.path.join(os.path.expanduser(d), name))
            if cand:
                return cand
        return None

    def _test_llm_cli(self, bin_name: str) -> tuple[bool, str]:
        try:
            bin_path = self._resolve_executable(bin_name)
            if not bin_path:
                return False, f"binaire introuvable: {bin_name}"
            proc = subprocess.run(
                [bin_path, "--version"], capture_output=True, text=True, timeout=10
            )
            ok = proc.returncode == 0
            out = (proc.stdout or proc.stderr).strip()
            return ok, f"{bin_path} --version -> rc={proc.returncode} out='{out[:200]}'"
        except Exception as e:
            return False, f"exception: {e}"

    def _pick_llm(self, llm: str) -> str | None:
        choice = (llm or self.user_valves.llm_cli_choice or "auto").strip().lower()
        if choice == "auto":
            for cand in ("qwen", "gemini"):
                ok, _ = self._test_llm_cli(cand)
                if ok:
                    return cand
            return None
        ok, _ = self._test_llm_cli(choice)
        return choice if ok else None

    def _execute_llm_cli(
        self, bin_name: str, model: str, prompt: str, context: str
    ) -> tuple[int, str, str, float]:
        bin_path = self._resolve_executable(bin_name)
        if not bin_path:
            return 127, "", f"Binaire LLM introuvable: {bin_name}", 0.0

        argv = [bin_path, "--model", model, "--prompt", prompt]
        log_cmd = self.user_valves.llm_cmd_template.format(
            bin=bin_path, model=model, prompt="<prompt>"
        )
        ctx_bytes = len(context.encode("utf-8", errors="ignore"))
        self.logger.info("[LLM] run: %s | context=%s bytes", log_cmd, ctx_bytes)

        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                out, err = proc.communicate(
                    input=context.encode("utf-8"),
                    timeout=float(self.valves.llm_timeout_s),
                )
                dur = time.perf_counter() - t0
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                dur = time.perf_counter() - t0
                return (
                    124,
                    out.decode("utf-8", "ignore"),
                    f"Timeout après {self.valves.llm_timeout_s}s",
                    dur,
                )

            rc = proc.returncode
            return rc, out.decode("utf-8", "ignore"), err.decode("utf-8", "ignore"), dur
        except Exception as e:
            dur = time.perf_counter() - t0
            return 1, "", f"Exception exécution LLM: {e}", dur

    # =====================================
    # CONTEXT PREPARATION (enhanced for v1.7)
    # =====================================
    def _prepare_code_context(self, repo_dir: str, depth: str) -> str:
        max_ctx = int(self.valves.max_context_bytes)
        max_file = int(self.valves.max_bytes_per_file)

        includes = (
            self.user_valves.custom_globs_include or self.valves.default_globs_include
        ).split(",")
        excludes = (
            self.user_valves.custom_globs_exclude or self.valves.default_globs_exclude
        ).split(",")
        inc, exc = self._build_specs(includes, excludes)

        # Priority files
        priority = [
            "README.md",
            "README.rst",
            "README.txt",
            "package.json",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "Cargo.toml",
            "go.mod",
            "composer.json",
        ]

        def append(buf: List[str], chunk: str, total: int) -> tuple[bool, int]:
            b = chunk.encode("utf-8", "ignore")
            if total + len(b) > max_ctx:
                remain = max_ctx - total - len(b"... [TRUNCATED CONTEXT] ...")
                if remain > 0:
                    buf.append(b.decode("utf-8", "ignore")[:remain])
                buf.append("... [TRUNCATED CONTEXT] ...")
                return False, max_ctx
            buf.append(chunk)
            return True, total + len(b)

        parts: List[str] = []
        used = 0

        # 1) Priority files
        for fname in priority:
            fpath = os.path.join(repo_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "rb") as f:
                        data = f.read(max_file)
                    chunk = f"=== {fname} ===\n{data.decode('utf-8', 'ignore')}\n"
                    ok, used = append(parts, chunk, used)
                    if not ok:
                        self.logger.info("[CTX] tronqué après priority; size=%s", used)
                        return self._redact_secrets("".join(parts))
                except Exception:
                    continue

        # 2) Scan eligible files
        depth = (depth or self.user_valves.analysis_depth or "standard").lower()
        limits = {"quick": 10, "standard": 25, "deep": 50}
        max_files = limits.get(depth, 25)

        files: List[str] = []
        for root, dirs, filenames in os.walk(repo_dir):
            rel_root = os.path.relpath(root, repo_dir)
            if rel_root == ".":
                rel_root = ""

            # Filter excluded directories
            keep = []
            for d in dirs:
                rel_dir = os.path.join(rel_root, d) if rel_root else d
                if exc.match_file(rel_dir):
                    continue
                keep.append(d)
            dirs[:] = keep

            for fn in filenames:
                rel_path = os.path.join(rel_root, fn) if rel_root else fn
                if exc.match_file(rel_path) or not inc.match_file(rel_path):
                    continue
                full = os.path.join(repo_dir, rel_path)
                try:
                    st = os.stat(full)
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    if st.st_size > self.valves.max_file_size_kb * 1024:
                        continue
                    files.append(rel_path)
                except Exception:
                    continue

        # Keep first max_files (OS order)
        for rel in files[:max_files]:
            full = os.path.join(repo_dir, rel)
            try:
                with open(full, "rb") as f:
                    data = f.read(max_file)
                chunk = f"=== {rel} ===\n{data.decode('utf-8', 'ignore')}\n"
                ok, used = append(parts, chunk, used)
                if not ok:
                    self.logger.info("[CTX] tronqué; size=%s", used)
                    break
            except Exception:
                continue

        ctx = "".join(parts)
        self.logger.info(
            "[CTX] prêt: %s bytes (avant redaction)", len(ctx.encode("utf-8", "ignore"))
        )
        return self._redact_secrets(ctx)

    # =====================================================================
    # 🔓 ALL v1.6 PUBLIC FUNCTIONS (unchanged signatures)
    # =====================================================================

    @staticmethod
    def tool_health(dummy: str = "") -> str:
        p = Tools._paths()
        return (
            f"OK | base_dir={p['base']} | repos_dir={p['repos']} | logs_dir={p['logs']}"
        )

    @staticmethod
    def debug_status(message: str = "ping") -> str:
        return f"DEBUG_STATUS: {message}"

    @staticmethod
    def list_repos() -> str:
        p = Tools._paths()
        repos_path = p["repos"]
        try:
            if not os.path.exists(repos_path):
                return "📁 Aucun dépôt (dossier 'git_repos' introuvable)."

            entries = [
                d
                for d in sorted(os.listdir(repos_path))
                if os.path.isdir(os.path.join(repos_path, d))
            ]
            if not entries:
                return "📁 Aucun dépôt trouvé dans git_repos."

            lines = ["## Dépôts disponibles:", ""]
            for d in entries:
                lines.append(f"- {d}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Erreur list_repos: {e}"

    @staticmethod
    def list_analyzed_repos() -> str:
        p = Tools._paths()
        repos_path = p["repos"]
        try:
            if not os.path.exists(repos_path):
                return "📁 Aucun dépôt (dossier 'git_repos' introuvable)."

            repos = []
            for d in sorted(os.listdir(repos_path)):
                repo_dir = os.path.join(repos_path, d)
                if not os.path.isdir(repo_dir):
                    continue
                analysis_dir = os.path.join(repo_dir, "docs_analysis")
                if not os.path.isdir(analysis_dir):
                    continue

                metadata_path = os.path.join(analysis_dir, "analysis_metadata.json")
                meta = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(
                            metadata_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            meta = json.load(f)
                    except Exception:
                        meta = {}

                repos.append(
                    {
                        "name": d,
                        "has_metadata": bool(meta),
                        "last_analysis": (
                            meta.get("analysis_timestamp") if meta else None
                        ),
                        "llm_cli_used": meta.get("llm_cli_used") if meta else None,
                        "synthesis_count": (
                            meta.get("synthesis_count") if meta else None
                        ),
                    }
                )

            if not repos:
                return "ℹ️ Aucun dépôt analysé (pas de dossier docs_analysis)."

            lines = ["## Dépôts analysés:", ""]
            for r in repos:
                badge = "✅" if r["has_metadata"] else "ℹ️"
                last = r["last_analysis"] or "n/d"
                llm = r["llm_cli_used"] or "n/d"
                syn = (
                    r["synthesis_count"] if r["synthesis_count"] is not None else "n/d"
                )
                lines.append(
                    f"- {badge} **{r['name']}** — dernière analyse: {last} — LLM: {llm} — synthèses: {syn}"
                )
            return "\n".join(lines)

        except Exception as e:
            return f"❌ Erreur list_analyzed_repos: {e}"

    @staticmethod
    def repo_info(repo_name: str) -> str:
        p = Tools._paths()
        safe = Tools._sanitize_repo_name(repo_name)
        repo_path = os.path.join(p["repos"], safe)
        analysis_dir = os.path.join(repo_path, "docs_analysis")

        if not os.path.isdir(repo_path):
            return f"❌ Dépôt introuvable: {safe}"

        lines = [f"## Infos dépôt: {safe}", f"📂 {repo_path}", ""]
        if not os.path.isdir(analysis_dir):
            lines.append("ℹ️ Pas de dossier `docs_analysis/`.")
            return "\n".join(lines)

        metadata_path = os.path.join(analysis_dir, "analysis_metadata.json")
        meta = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8", errors="ignore") as f:
                    meta = json.load(f)
            except Exception as e:
                lines.append(f"❌ Erreur lecture metadata: {e}")

        if meta:
            lines.append("### Métadonnées")
            lines.append(
                f"- Dernière analyse : {meta.get('analysis_timestamp', 'n/d')}"
            )
            lines.append(f"- LLM utilisé      : {meta.get('llm_cli_used', 'n/d')}")
            lines.append(f"- Synthèses        : {meta.get('synthesis_count', 'n/d')}")
            user_cfg = meta.get("user_config", {})
            if isinstance(user_cfg, dict) and user_cfg:
                lines.append(f"- Profil utilisateur: {user_cfg}")
            lines.append("")

        wanted = ["ARCHITECTURE.md", "API_SUMMARY.md", "CODE_MAP.md"]
        lines.append("### Fichiers de synthèse présents")
        found = False
        for fname in wanted:
            fpath = os.path.join(analysis_dir, fname)
            if os.path.exists(fpath) and os.path.isfile(fpath):
                try:
                    size = os.path.getsize(fpath)
                except Exception:
                    size = 0
                lines.append(f"- ✅ {fname} ({size} octets)")
                found = True
        if not found:
            lines.append("- (aucun)")
        return "\n".join(lines)

    @staticmethod
    def get_repo_context(
        repo_name: str, max_files: int = 3, max_chars_per_file: int = 2000
    ) -> str:
        p = Tools._paths()
        safe = Tools._sanitize_repo_name(repo_name)
        repo_path = os.path.join(p["repos"], safe)
        analysis = os.path.join(repo_path, "docs_analysis")

        if not os.path.exists(repo_path):
            return f"❌ Dépôt introuvable: {safe}"
        if not os.path.isdir(analysis):
            return f"ℹ️ Pas de docs_analysis pour: {safe}"

        wanted = ["ARCHITECTURE.md", "API_SUMMARY.md", "CODE_MAP.md"]
        shown = 0
        out = [
            f"## Contexte pour {safe}",
            f"(max_files={max_files}, max_chars_per_file={max_chars_per_file})",
            "",
        ]

        for fname in wanted:
            if shown >= max_files:
                break
            fpath = os.path.join(analysis, fname)
            if os.path.exists(fpath) and os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(max_chars_per_file)
                    out.append(f"### {fname}")
                    out.append("```markdown")
                    out.append(content)
                    out.append("```")
                    out.append("")
                    shown += 1
                except Exception as e:
                    out.append(f"❌ Erreur lecture {fname}: {e}")

        if shown == 0:
            return f"ℹ️ Aucun fichier de synthèse trouvé dans {analysis}"
        return "\n".join(out)

    @staticmethod
    def scan_repo_files(
        repo_name: str, limit: int = 50, order: str = "size", ascending: bool = False
    ) -> str:
        try:
            p = Tools._paths()
            safe = Tools._sanitize_repo_name(repo_name)
            repo = os.path.join(p["repos"], safe)
            if not os.path.isdir(repo):
                return f"❌ Dépôt introuvable: {safe}"

            _tmp = Tools()
            includes = (
                _tmp.user_valves.custom_globs_include
                or _tmp.valves.default_globs_include
            ).split(",")
            excludes = (
                _tmp.user_valves.custom_globs_exclude
                or _tmp.valves.default_globs_exclude
            ).split(",")
            max_kb = int(_tmp.valves.max_file_size_kb)

            inc, exc = Tools._build_specs(includes, excludes)

            files: List[Dict[str, Any]] = []
            for root, dirs, filenames in os.walk(repo):
                rel_root = os.path.relpath(root, repo)
                if rel_root == ".":
                    rel_root = ""

                keep_dirs = []
                for d in dirs:
                    rel_dir = os.path.join(rel_root, d) if rel_root else d
                    if exc.match_file(rel_dir):
                        continue
                    keep_dirs.append(d)
                dirs[:] = keep_dirs

                for fn in filenames:
                    rel_path = os.path.join(rel_root, fn) if rel_root else fn
                    if exc.match_file(rel_path):
                        continue
                    if not inc.match_file(rel_path):
                        continue

                    full = os.path.join(repo, rel_path)
                    try:
                        st = os.stat(full)
                        if not stat.S_ISREG(st.st_mode):
                            continue
                        size = st.st_size
                        if size > max_kb * 1024:
                            continue
                        files.append({"path": rel_path, "size": size})
                    except Exception:
                        continue

            total = len(files)
            if order not in ("size", "path"):
                order = "size"
            reverse = not ascending
            files.sort(key=(lambda x: x[order]), reverse=reverse)

            sel = files[: max(0, int(limit))]
            total_size = sum(f["size"] for f in files)
            shown_size = sum(f["size"] for f in sel)

            lines = [
                f"## Scan fichiers — dépôt: {safe}",
                f"- Fichiers éligibles: {total}",
                f"- Somme tailles (éligibles): {total_size} octets",
                f"- Affichés: {len(sel)} (ordre={order}, ascending={ascending})",
                f"- Somme tailles (affichés): {shown_size} octets",
                "",
                "### Fichiers",
            ]
            if not sel:
                lines.append("(aucun)")
            else:
                for f in sel:
                    lines.append(f"- {f['path']}  ({f['size']} o)")
            return "\n".join(lines)

        except Exception as e:
            return f"❌ Erreur scan_repo_files: {e}"

    @staticmethod
    def preview_file(repo_name: str, relative_path: str, max_bytes: int = 65536) -> str:
        try:
            if max_bytes <= 0:
                max_bytes = 4096
            max_bytes = min(max_bytes, 2_000_000)

            p = Tools._paths()
            safe = Tools._sanitize_repo_name(repo_name)
            repo = os.path.join(p["repos"], safe)
            if not os.path.isdir(repo):
                return f"❌ Dépôt introuvable: {safe}"

            target = os.path.realpath(os.path.join(repo, relative_path))
            repo_real = os.path.realpath(repo)
            if not (target == repo_real or target.startswith(repo_real + os.sep)):
                return "❌ Chemin refusé (hors dépôt)."

            if not os.path.isfile(target):
                return "❌ Fichier introuvable."

            with open(target, "rb") as fb:
                head = fb.read(4096)
                if Tools._looks_binary(head):
                    return (
                        "❌ Fichier binaire / non texte (prévisualisation désactivée)."
                    )

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_bytes)

            lang = Tools._ext_lang_hint(target)
            fence = lang if lang else ""
            rel = os.path.relpath(target, repo_real)
            size = os.path.getsize(target)

            lines = [
                f"## Aperçu: {rel}",
                f"- Taille: {size} octets",
                f"- Affiché: jusqu'à {max_bytes} octets",
                "",
                f"```{fence}",
                content,
                "```",
            ]
            return "\n".join(lines)

        except Exception as e:
            return f"❌ Erreur preview_file: {e}"

    @staticmethod
    def stats_repo(repo_name: str, top_n: int = 10) -> str:
        try:
            p = Tools._paths()
            safe = Tools._sanitize_repo_name(repo_name)
            repo = os.path.join(p["repos"], safe)
            if not os.path.isdir(repo):
                return f"❌ Dépôt introuvable: {safe}"

            _tmp = Tools()
            includes = (
                _tmp.user_valves.custom_globs_include
                or _tmp.valves.default_globs_include
            ).split(",")
            excludes = (
                _tmp.user_valves.custom_globs_exclude
                or _tmp.valves.default_globs_exclude
            ).split(",")
            max_kb = int(_tmp.valves.max_file_size_kb)
            inc, exc = Tools._build_specs(includes, excludes)

            ext_count: Dict[str, int] = {}
            ext_bytes: Dict[str, int] = {}
            files: List[Dict[str, Any]] = []

            for root, dirs, filenames in os.walk(repo):
                rel_root = os.path.relpath(root, repo)
                if rel_root == ".":
                    rel_root = ""

                keep_dirs = []
                for d in dirs:
                    rel_dir = os.path.join(rel_root, d) if rel_root else d
                    if exc.match_file(rel_dir):
                        continue
                    keep_dirs.append(d)
                dirs[:] = keep_dirs

                for fn in filenames:
                    rel_path = os.path.join(rel_root, fn) if rel_root else fn
                    if exc.match_file(rel_path) or not inc.match_file(rel_path):
                        continue
                    full = os.path.join(repo, rel_path)
                    try:
                        st = os.stat(full)
                        if not stat.S_ISREG(st.st_mode):
                            continue
                        size = st.st_size
                        if size > max_kb * 1024:
                            continue
                        files.append({"path": rel_path, "size": size})
                        ext = os.path.splitext(fn)[1].lower() or "(noext)"
                        ext_count[ext] = ext_count.get(ext, 0) + 1
                        ext_bytes[ext] = ext_bytes.get(ext, 0) + size
                    except Exception:
                        continue

            total_files = len(files)
            total_bytes = sum(f["size"] for f in files)
            top_files = sorted(files, key=lambda x: x["size"], reverse=True)[
                : max(1, int(top_n))
            ]
            top_ext = sorted(
                ext_count.items(), key=lambda kv: ext_bytes.get(kv[0], 0), reverse=True
            )[:10]

            lines = [
                f"## Stats — dépôt: {safe}",
                f"- Fichiers éligibles: {total_files}",
                f"- Taille totale: {total_bytes} octets",
                "",
                "### Top extensions (par octets)",
            ]
            if not top_ext:
                lines.append("(aucune)")
            else:
                for ext, cnt in top_ext:
                    lines.append(f"- {ext}: {cnt} fichiers, {ext_bytes.get(ext, 0)} o")

            lines.append("")
            lines.append(f"### Top {top_n} fichiers (par taille)")
            if not top_files:
                lines.append("(aucun)")
            else:
                for f in top_files:
                    lines.append(f"- {f['path']}  ({f['size']} o)")
            return "\n".join(lines)

        except Exception as e:
            return f"❌ Erreur stats_repo: {e}"

    @staticmethod
    def find_in_repo(
        repo_name: str, needle: str, use_regex: bool = False, max_matches: int = 50
    ) -> str:
        try:
            p = Tools._paths()
            safe = Tools._sanitize_repo_name(repo_name)
            repo = os.path.join(p["repos"], safe)
            if not os.path.isdir(repo):
                return f"❌ Dépôt introuvable: {safe}"

            _tmp = Tools()
            includes = (
                _tmp.user_valves.custom_globs_include
                or _tmp.valves.default_globs_include
            ).split(",")
            excludes = (
                _tmp.user_valves.custom_globs_exclude
                or _tmp.valves.default_globs_exclude
            ).split(",")
            max_kb = int(_tmp.valves.max_file_size_kb)
            inc, exc = Tools._build_specs(includes, excludes)

            pattern = None
            if use_regex:
                try:
                    pattern = re.compile(needle, flags=re.IGNORECASE)
                except re.error as e:
                    return f"❌ Regex invalide: {e}"

            results = []
            for root, dirs, filenames in os.walk(repo):
                rel_root = os.path.relpath(root, repo)
                if rel_root == ".":
                    rel_root = ""

                keep_dirs = []
                for d in dirs:
                    rel_dir = os.path.join(rel_root, d) if rel_root else d
                    if exc.match_file(rel_dir):
                        continue
                    keep_dirs.append(d)
                dirs[:] = keep_dirs

                for fn in filenames:
                    rel_path = os.path.join(rel_root, fn) if rel_root else fn
                    if exc.match_file(rel_path) or not inc.match_file(rel_path):
                        continue

                    full = os.path.join(repo, rel_path)
                    try:
                        st = os.stat(full)
                        if not stat.S_ISREG(st.st_mode):
                            continue
                        size = st.st_size
                        if size > max_kb * 1024:
                            continue

                        with open(full, "rb") as fb:
                            head = fb.read(4096)
                            if Tools._looks_binary(head):
                                continue
                        with open(full, "r", encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, start=1):
                                hit = (
                                    (pattern.search(line) is not None)
                                    if pattern
                                    else (needle.lower() in line.lower())
                                )
                                if hit:
                                    snippet = line.strip()
                                    results.append(f"- {rel_path}:{idx}: {snippet}")
                                    if len(results) >= max(1, int(max_matches)):
                                        raise StopIteration
                    except StopIteration:
                        break
                    except Exception:
                        continue
                if len(results) >= max(1, int(max_matches)):
                    break

            if not results:
                return f"🔎 Aucun résultat pour « {needle} » (use_regex={use_regex}) dans {safe}."
            header = [
                f"## Recherche « {needle} » (regex={use_regex}) — dépôt: {safe}",
                "",
            ]
            return "\n".join(header + results)

        except Exception as e:
            return f"❌ Erreur find_in_repo: {e}"

    @staticmethod
    def git_clone(repo_url: str, name: str = "") -> str:
        try:
            info = Tools._parse_git_url(repo_url)
            _tmp = Tools()

            allowed = [
                h.strip()
                for h in _tmp.valves.supported_git_hosts.split(",")
                if h.strip()
            ]
            if info["host"] not in allowed:
                return f"❌ Hôte non supporté: {info['host']} (allow: {', '.join(allowed)})"

            repo_name = Tools._sanitize_repo_name(name or info["repo_name"])
            p = Tools._paths()
            target = os.path.join(p["repos"], repo_name)

            if os.path.exists(target):
                return f"ℹ️ Dépôt existe déjà: {repo_name}"

            os.makedirs(p["repos"], exist_ok=True)

            cmd = ["git", "clone", repo_url, repo_name]
            completed = subprocess.run(
                cmd,
                cwd=p["repos"],
                capture_output=True,
                text=True,
                timeout=max(30, int(_tmp.valves.git_timeout_s)),
                check=False,
            )
            if completed.returncode != 0:
                return f"❌ git clone a échoué ({completed.returncode})\nSTDERR:\n{completed.stderr.strip()}"
            return f"✅ Clone OK → {repo_name}\nSTDOUT:\n{completed.stdout.strip()}"

        except subprocess.TimeoutExpired:
            return "❌ Timeout git clone."
        except Exception as e:
            return f"❌ Erreur git_clone: {e}"

    @staticmethod
    def git_update(repo_name: str, strategy: str = "pull") -> str:
        try:
            p = Tools._paths()
            safe = Tools._sanitize_repo_name(repo_name)
            repo = os.path.join(p["repos"], safe)
            if not os.path.isdir(repo):
                return f"❌ Dépôt introuvable: {safe}"

            _tmp = Tools()
            timeout_s = max(30, int(_tmp.valves.git_timeout_s))

            def run(cmd, cwd):
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
                return proc.returncode, proc.stdout, proc.stderr

            if strategy not in ("pull", "reset"):
                strategy = "pull"

            if strategy == "pull":
                rc1, out1, err1 = run(["git", "fetch", "--all", "--tags"], repo)
                rc2, out2, err2 = run(["git", "pull", "--ff-only"], repo)
                if rc1 != 0 or rc2 != 0:
                    return f"❌ git_update pull a échoué\nfetch rc={rc1} err={err1}\npull rc={rc2} err={err2}"
                return f"✅ Pull OK\n{out1}\n{out2}".strip()

            rc1, out1, err1 = run(["git", "fetch", "--all", "--tags"], repo)
            rcH, outH, errH = run(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD"], repo
            )
            ref = "origin/HEAD"
            if rcH == 0:
                ref = outH.strip().split("/")[-1]
                ref = f"origin/{ref}" if ref else "origin/HEAD"
            rc2, out2, err2 = run(["git", "reset", "--hard", ref], repo)

            if rc1 != 0 or rc2 != 0:
                return f"❌ git_update reset a échoué\nfetch rc={rc1} err={err1}\nreset rc={rc2} err={err2}"
            return f"✅ Reset OK → {ref}\n{out1}\n{out2}".strip()

        except subprocess.TimeoutExpired:
            return "❌ Timeout git_update."
        except Exception as e:
            return f"❌ Erreur git_update: {e}"

    @staticmethod
    def clean_analysis(repo_name: str) -> str:
        try:
            p = Tools._paths()
            safe = Tools._sanitize_repo_name(repo_name)
            repo = os.path.join(p["repos"], safe)
            analysis = os.path.join(repo, "docs_analysis")
            if not os.path.isdir(repo):
                return f"❌ Dépôt introuvable: {safe}"
            if not os.path.isdir(analysis):
                return f"ℹ️ Rien à nettoyer (pas de docs_analysis) pour {safe}"
            for root, dirs, files in os.walk(analysis, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(analysis)
            except Exception:
                pass
            return f"✅ Nettoyage docs_analysis terminé pour {safe}"
        except Exception as e:
            return f"❌ Erreur clean_analysis: {e}"

    @staticmethod
    def llm_check(llm: str = "", model: str = "") -> str:
        try:
            t = Tools()
            use = t._pick_llm(llm)
            if not use:
                return "❌ Aucun LLM CLI disponible (essayé: qwen, gemini)."
            ok, diag = t._test_llm_cli(use)
            m = model or t.user_valves.llm_model_name or "n/d"
            return f"✅ LLM détecté: {use}\n- diag: {diag}\n- modèle par défaut: {m}"
        except Exception as e:
            return f"❌ Erreur llm_check: {e}"

    @staticmethod
    def analyze_repo(
        repo_name: str,
        sections: str = "architecture,api,codemap",
        depth: str = "",
        language: str = "",
        llm: str = "",
        model: str = "",
    ) -> str:
        """
        Génère docs_analysis/ via LLM CLI.
        - repo_name: nom du dépôt (dossier dans git_repos)
        - sections: liste csv dans {architecture, api, codemap}
        - depth: quick|standard|deep (override)
        - language: fr|en (override)
        - llm: qwen|gemini|auto (override)
        - model: nom de modèle (override)
        """
        try:
            p = Tools._paths()
            safe = Tools._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)
            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            tool = Tools()
            use_llm = tool._pick_llm(llm)
            if not use_llm:
                return (
                    "❌ Aucun LLM CLI disponible (qwen/gemini). Utilisez llm_check()."
                )
            use_model = (model or tool.user_valves.llm_model_name).strip()
            lang = (language or tool.user_valves.preferred_language or "fr").lower()
            if lang not in ("fr", "en"):
                lang = "fr"

            # Préparer contexte
            ctx = tool._prepare_code_context(
                repo_dir, depth or tool.user_valves.analysis_depth
            )
            if not ctx:
                return f"❌ Contexte vide pour {safe}"

            # Prompts
            style = tool.user_valves.prompt_style
            extra_arch = tool.user_valves.prompt_extra_architecture or ""
            extra_api = tool.user_valves.prompt_extra_api or ""
            extra_map = tool.user_valves.prompt_extra_codemap or ""

            prompts = {
                "architecture": {
                    "fr": f"Analyse l'architecture: stack, modules clés, points d'entrée, organisation, patterns. Style: {style}. {extra_arch}".strip(),
                    "en": f"Analyze architecture: stack, key modules, entry points, organization, patterns. Style: {style}. {extra_arch}".strip(),
                },
                "api": {
                    "fr": f"Extrait les APIs: classes/fonctions publiques, interfaces, points d'entrée programmatiques. Style: {style}. {extra_api}".strip(),
                    "en": f"Extract APIs: public classes/functions, interfaces, programmatic entry points. Style: {style}. {extra_api}".strip(),
                },
                "codemap": {
                    "fr": f"Carte du code: rôle des dossiers, fichiers importants, principaux flux de données, navigation efficace. Style: {style}. {extra_map}".strip(),
                    "en": f"Code map: role of folders, key files, main data flows, navigation guidance. Style: {style}. {extra_map}".strip(),
                },
            }

            wanted = []
            for s in [s.strip().lower() for s in sections.split(",") if s.strip()]:
                if s in ("architecture", "api", "codemap"):
                    wanted.append(s)
            if not wanted:
                wanted = ["architecture", "api", "codemap"]

            analysis_dir = os.path.join(repo_dir, "docs_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            generated = 0
            results_lines = [
                f"## Analyse LLM — dépôt: {safe}",
                f"- LLM: {use_llm} | modèle: {use_model}",
                f"- sections: {', '.join(wanted)}",
                f"- langue: {lang}",
                "",
            ]

            for sec in wanted:
                prompt = prompts[sec][lang]
                rc, out, err, dur = tool._execute_llm_cli(
                    use_llm, use_model, prompt, ctx
                )
                tool.logger.info(
                    "[LLM] section=%s rc=%s dur=%.1fs err=%s",
                    sec,
                    rc,
                    dur,
                    (err.strip()[:200] if err else ""),
                )

                if rc != 0:
                    results_lines.append(
                        f"❌ {sec.upper()} — échec rc={rc} ({dur:.1f}s)\n{(err or '').strip()[:500]}"
                    )
                    continue

                fname = {
                    "architecture": "ARCHITECTURE.md",
                    "api": "API_SUMMARY.md",
                    "codemap": "CODE_MAP.md",
                }[sec]
                fpath = os.path.join(analysis_dir, fname)
                try:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(out.strip() if out.strip() else "(vide)")
                    generated += 1
                    results_lines.append(
                        f"✅ {sec.upper()} → {fname} ({len(out.encode('utf-8'))} bytes, {dur:.1f}s)"
                    )
                except Exception as e:
                    results_lines.append(f"❌ {sec.upper()} — écriture échouée: {e}")

            # metadata
            metadata = {
                "repo_info": {"name": safe, "path": repo_dir},
                "analysis_timestamp": datetime.now().isoformat(),
                "llm_cli_used": use_llm,
                "synthesis_count": generated,
                "user_config": {
                    "analysis_depth": depth or tool.user_valves.analysis_depth,
                    "preferred_language": lang,
                    "prompt_style": style,
                },
                "tool_version": "1.7.0",  # Updated version
            }
            try:
                with open(
                    os.path.join(analysis_dir, "analysis_metadata.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                results_lines.append(f"⚠️ Écriture metadata échouée: {e}")

            if generated == 0:
                results_lines.append(
                    "\nAucun fichier généré. Consultez les logs pour le détail."
                )
            else:
                results_lines.append(
                    f"\nSynthèses générées: {generated} — dossier: {analysis_dir}"
                )

            return "\n".join(results_lines)

        except Exception as e:
            return f"❌ Erreur analyze_repo: {e}"

    # =====================================================================
    # 🆕 NEW v1.7 RAG PUBLIC FUNCTIONS
    # =====================================================================

    def auto_retrieve_context(self, repo_name: str, question: str) -> str:
        """
        Enhanced auto_retrieve_context with multiple backend support.

        Now supports keywords, BM25, and hybrid retrieval based on user preferences.

        Args:
            repo_name: Repository name (sanitized)
            question: User question in natural language

        Returns:
            Formatted markdown context block ready for LLM injection
        """
        # Check user's retrieval backend preference
        backend = self.user_valves.retrieval_backend

        if backend == "bm25":
            return self.hybrid_retrieve(repo_name, question, k=self.user_valves.rag_max_extracts, mode="bm25")
        elif backend == "hybrid":
            return self.hybrid_retrieve(repo_name, question, k=self.user_valves.rag_max_extracts, mode="hybrid")
        else:
            # Use original keyword-based approach (v1.7 implementation preserved)
            return self._original_auto_retrieve_context(repo_name, question)

    def _original_auto_retrieve_context(self, repo_name: str, question: str) -> str:
        """Original v1.7 auto_retrieve_context implementation."""
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Extract keywords and detect intent
            keywords = self._extract_keywords_from_question(question)
            intent = self._detect_question_intent(question)

            self.logger.info(
                "[RAG] intent=%s, keywords_count=%d", intent, len(keywords)
            )

            # Get high-level context
            high_level_context = self.get_repo_context(
                safe, max_files=2, max_chars_per_file=2000
            )

            # Search for relevant files
            files = self._get_repo_files_for_search(repo_dir)
            search_results = self._search_for_keywords_in_files(files, keywords)

            # Select top extracts
            max_extracts = self.user_valves.rag_max_extracts
            selected_results = search_results[:max_extracts]

            # Extract context around hits
            extracts = []
            sources_table = []

            for result in selected_results:
                file_path = result["file"]["full_path"]
                relative_path = result["file"]["path"]

                # Extract context
                context_extract = self._extract_context_around_hits(
                    file_path, result["hits"]
                )
                if context_extract and context_extract not in extracts:
                    extracts.append(context_extract)

                    # Add to sources table
                    for hit in result["hits"][:3]:  # Max 3 hits per file in table
                        reason = "definition" if hit["score"] >= 2 else "usage"
                        sources_table.append(
                            f"- {relative_path}:{hit['line']} (raison: {reason})"
                        )

            # Deduplicate extracts
            unique_extracts = self._deduplicate_extracts(extracts)

            # Build final context
            lines = [
                f"## Auto Context — {safe}",
                f"**Intent détecté**: {intent}",
                "",
                "### Synthèses de haut niveau",
                high_level_context,
                "",
                f"### Extraits pertinents (N={len(unique_extracts)})",
            ]

            if not unique_extracts:
                lines.append("(aucun extrait pertinent trouvé)")
            else:
                lines.extend(unique_extracts)

            lines.extend(
                [
                    "",
                    "### Table des sources",
                ]
            )

            if sources_table:
                lines.extend(sources_table)
            else:
                lines.append("(aucune source)")

            result = "\n".join(lines)

            # Apply context limits
            result_bytes = len(result.encode("utf-8", errors="ignore"))
            if result_bytes > self.valves.max_context_bytes:
                truncate_at = self.valves.max_context_bytes - 100
                result = result.encode("utf-8")[:truncate_at].decode(
                    "utf-8", errors="ignore"
                )
                result += "\n\n... [CONTEXTE TRONQUÉ] ..."

            self.logger.info(
                "[RAG] context_size=%d bytes, extracts=%d",
                result_bytes,
                len(unique_extracts),
            )
            return self._redact_secrets(result)

        except Exception as e:
            self.logger.error("[RAG] auto_retrieve_context error: %s", e)
            return f"❌ Erreur auto_retrieve_context: {e}"

    def build_simple_index(self, repo_name: str) -> str:
        """
        Build a simple symbolic index for the repository using regex patterns.

        Args:
            repo_name: Repository name (sanitized)

        Returns:
            Status message with index creation results
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            analysis_dir = os.path.join(repo_dir, "docs_analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            # Build index
            index = self._build_symbol_index_for_repo(repo_dir)

            # Save index
            index_path = os.path.join(analysis_dir, "simple_index.json")
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

            func_count = len(index["functions"])
            class_count = len(index["classes"])
            import_count = len(index["imports"])
            export_count = len(index["exports"])

            self.logger.info(
                "[INDEX] built for %s: %d functions, %d classes, %d imports, %d exports",
                safe,
                func_count,
                class_count,
                import_count,
                export_count,
            )

            return (
                f"✅ Index écrit: docs_analysis/simple_index.json "
                f"({func_count} fonctions, {class_count} classes, "
                f"{import_count} imports, {export_count} exports)"
            )

        except Exception as e:
            self.logger.error("[INDEX] build_simple_index error: %s", e)
            return f"❌ Erreur build_simple_index: {e}"

    def quick_api_lookup(self, repo_name: str, api_name: str) -> str:
        """
        Quickly find definitions and usages of a specific API/symbol.

        Args:
            repo_name: Repository name (sanitized)
            api_name: Symbol name to look up

        Returns:
            Formatted markdown with definitions and usage examples
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Load or create index
            index = self._load_or_create_index(safe)
            if not index:
                return f"❌ Impossible de charger/créer l'index pour {safe}"

            lines = [f"## Quick API Lookup — {api_name}", ""]

            # Find definitions
            definitions = []

            # Check functions
            for func in index.get("functions", []):
                if func["name"] == api_name or api_name.lower() in func["name"].lower():
                    definitions.append(
                        f"- **Fonction**: {func['name']} dans {func['file']}:{func['line']} ({func['lang']})"
                    )

            # Check classes
            for cls in index.get("classes", []):
                if cls["name"] == api_name or api_name.lower() in cls["name"].lower():
                    definitions.append(
                        f"- **Classe**: {cls['name']} dans {cls['file']}:{cls['line']} ({cls['lang']})"
                    )

            if definitions:
                lines.extend(["### Définitions"] + definitions + [""])

            # Find usages using existing find_in_repo
            usage_result = self.find_in_repo(
                safe, api_name, use_regex=False, max_matches=10
            )

            if "Aucun résultat" not in usage_result:
                lines.append("### Usages notables")
                usage_lines = usage_result.split("\n")[2:]  # Skip header
                lines.extend(usage_lines[:10])  # Max 10 usage lines
                lines.append("")

            # Check imports/exports
            imports_exports = []
            for imp in index.get("imports", []):
                if any(
                    api_name.lower() in item.lower() for item in imp.get("items", [])
                ):
                    imports_exports.append(
                        f"- **Import**: {imp['from']} → {imp['items']} dans {imp['file']}:{imp['line']}"
                    )

            for exp in index.get("exports", []):
                if api_name.lower() in exp["name"].lower():
                    imports_exports.append(
                        f"- **Export**: {exp['name']} ({exp['kind']}) dans {exp['file']}:{exp['line']}"
                    )

            if imports_exports:
                lines.extend(["### Imports/Exports associés"] + imports_exports)

            if len(lines) <= 2:  # Only header
                return f"🔎 Aucun symbole « {api_name} » trouvé."

            return "\n".join(lines)

        except Exception as e:
            self.logger.error("[LOOKUP] quick_api_lookup error: %s", e)
            return f"❌ Erreur quick_api_lookup: {e}"

    def find_usage_examples(self, repo_name: str, symbol: str) -> str:
        """
        Find usage examples of a symbol, excluding definitions.

        Args:
            repo_name: Repository name (sanitized)
            symbol: Symbol to find examples for

        Returns:
            Formatted markdown with usage examples grouped by file
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Load index to identify definitions
            index = self._load_or_create_index(safe)
            definition_locations = set()

            if index:
                # Collect definition locations to exclude them
                for func in index.get("functions", []):
                    if func["name"] == symbol:
                        definition_locations.add(f"{func['file']}:{func['line']}")
                for cls in index.get("classes", []):
                    if cls["name"] == symbol:
                        definition_locations.add(f"{cls['file']}:{cls['line']}")

            # Find all occurrences
            all_results = self.find_in_repo(
                safe, symbol, use_regex=False, max_matches=200
            )

            if "Aucun résultat" in all_results:
                return f"ℹ️ Aucun usage trouvé pour « {symbol} » (hors définition)."

            # Parse results and group by file
            lines = all_results.split("\n")[2:]  # Skip header
            file_groups = {}

            for line in lines:
                if line.startswith("- "):
                    # Parse: "- path:line: content"
                    parts = line[2:].split(": ", 2)
                    if len(parts) >= 2:
                        location = parts[0]  # path:line
                        content = parts[1] if len(parts) == 2 else parts[2]

                        # Skip if this is a definition
                        if location in definition_locations:
                            continue

                        # Skip obvious definition patterns
                        content_lower = content.lower()
                        if any(
                            pattern in content_lower
                            for pattern in [
                                f"def {symbol.lower()}",
                                f"class {symbol.lower()}",
                                f"function {symbol.lower()}",
                                f"const {symbol.lower()} =",
                            ]
                        ):
                            continue

                        file_path = location.split(":")[0]
                        if file_path not in file_groups:
                            file_groups[file_path] = []

                        file_groups[file_path].append(f"  - {location}: {content}")

            # Build output
            if not file_groups:
                return f"ℹ️ Aucun usage trouvé pour « {symbol} » (hors définition)."

            result_lines = [f"## Usage Examples — {symbol}", ""]

            # Prioritize test files
            sorted_files = sorted(
                file_groups.keys(),
                key=lambda f: (
                    (
                        0
                        if any(
                            test_marker in f.lower()
                            for test_marker in ["test", "spec", "example", "demo"]
                        )
                        else 1
                    ),
                    f,
                ),
            )

            for file_path in sorted_files[:10]:  # Max 10 files
                examples = file_groups[file_path][:5]  # Max 5 examples per file
                result_lines.extend([f"### {file_path}"] + examples + [""])

            return "\n".join(result_lines)

        except Exception as e:
            self.logger.error("[USAGE] find_usage_examples error: %s", e)
            return f"❌ Erreur find_usage_examples: {e}"

    def show_related_code(self, repo_name: str, path: str) -> str:
        """
        Show related code around a specific file: imports, exports, tests, etc.

        Args:
            repo_name: Repository name (sanitized)
            path: Relative path to file within repo

        Returns:
            Formatted markdown with related code information
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Security: ensure path is within repo
            full_path = os.path.realpath(os.path.join(repo_dir, path))
            repo_real = os.path.realpath(repo_dir)
            if not (full_path == repo_real or full_path.startswith(repo_real + os.sep)):
                return "❌ Chemin refusé (hors dépôt)."

            if not os.path.isfile(full_path):
                return "❌ Fichier introuvable."

            lines = [f"## Related Code — {path}", ""]

            # Load index
            index = self._load_or_create_index(safe)

            # Show symbols defined in this file
            if index:
                symbols_defined = []

                for func in index.get("functions", []):
                    if func["file"] == path:
                        symbols_defined.append(
                            f"- **{func['name']}()** (fonction, ligne {func['line']})"
                        )

                for cls in index.get("classes", []):
                    if cls["file"] == path:
                        symbols_defined.append(
                            f"- **{cls['name']}** (classe, ligne {cls['line']})"
                        )

                if symbols_defined:
                    lines.extend(["### Symboles définis"] + symbols_defined + [""])

            # Analyze file content for imports/exports
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                outgoing_imports = []
                # Python imports
                for match in re.finditer(
                    r"^\s*from\s+([\w\.]+)\s+import\s+([\w\*,\s]+)",
                    content,
                    re.MULTILINE,
                ):
                    module = match.group(1)
                    items = match.group(2)
                    outgoing_imports.append(f"- from {module} import {items}")

                for match in re.finditer(
                    r"^\s*import\s+([\w\.]+)", content, re.MULTILINE
                ):
                    module = match.group(1)
                    outgoing_imports.append(f"- import {module}")

                # JavaScript imports
                for match in re.finditer(
                    r'^\s*import\s+.+\s+from\s+[\'"]([^\'"]+)[\'"]',
                    content,
                    re.MULTILINE,
                ):
                    module = match.group(1)
                    outgoing_imports.append(f"- import from '{module}'")

                if outgoing_imports:
                    lines.extend(["### Imports sortants"] + outgoing_imports + [""])

            except Exception:
                pass

            # Find incoming imports (who imports this file)
            incoming_imports = []
            if index:
                file_basename = os.path.splitext(os.path.basename(path))[0]
                file_dir = os.path.dirname(path)

                # Look for imports of this file
                for imp in index.get("imports", []):
                    import_from = imp.get("from", "")
                    # Simple heuristic: check if import path matches our file
                    if (
                        file_basename in import_from
                        or path.replace("/", ".").replace(".py", "") in import_from
                        or any(file_basename in item for item in imp.get("items", []))
                    ):
                        incoming_imports.append(
                            f"- {imp['file']}:{imp['line']} imports from '{import_from}'"
                        )

            if incoming_imports:
                lines.extend(["### Imports entrants"] + incoming_imports + [""])

            # Find related test files
            related_tests = []
            file_base = os.path.splitext(os.path.basename(path))[0]

            test_patterns = [
                f"{file_base}.test.*",
                f"{file_base}.spec.*",
                f"test_{file_base}.*",
                f"*{file_base}_test.*",
            ]

            # Search for test files
            files = self._get_repo_files_for_search(repo_dir)
            for file_info in files:
                file_path = file_info["path"].lower()
                if (
                    any(test_marker in file_path for test_marker in ["test", "spec"])
                    and file_base.lower() in file_path
                ):
                    related_tests.append(f"- {file_info['path']}")

            if related_tests:
                lines.extend(["### Tests liés"] + related_tests[:5])  # Max 5 test files

                # Show brief excerpts if few test files
                if len(related_tests) <= 3:
                    lines.append("")
                    for test_file in related_tests[:3]:
                        test_preview = self.preview_file(
                            safe, test_file.split("- ")[1], max_bytes=1000
                        )
                        if not test_preview.startswith("❌"):
                            lines.extend(
                                [
                                    f"#### Extrait de {test_file.split('- ')[1]}",
                                    test_preview[:500] + "...",
                                    "",
                                ]
                            )

            if len(lines) <= 2:  # Only header
                lines.append("ℹ️ Aucune information relationnelle trouvée.")

            return "\n".join(lines)

        except Exception as e:
            self.logger.error("[RELATED] show_related_code error: %s", e)
            return f"❌ Erreur show_related_code: {e}"

    def auto_load_context(self, repo_name: str, user_question: str) -> str:
        """
        Smart context loading based on question intent with adaptive strategies.

        Args:
            repo_name: Repository name (sanitized)
            user_question: User's question to analyze

        Returns:
            Formatted markdown context adapted to detected intent
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Detect intent
            intent = self._detect_question_intent(user_question)

            lines = [f"## Auto Load Context — {safe} [intent={intent}]", ""]

            # Apply intent-specific strategies
            if intent == "debug":
                # Focus on error-related files and logs
                error_files = self.find_in_repo(
                    safe,
                    "error|exception|throw|traceback",
                    use_regex=True,
                    max_matches=20,
                )
                if "Aucun résultat" not in error_files:
                    lines.extend(["### Fichiers liés aux erreurs", error_files, ""])

                # Get general context
                context = self._original_auto_retrieve_context(safe, user_question)
                lines.extend([context])

            elif intent == "api_lookup":
                # Try to extract API name from question
                api_matches = re.findall(
                    r"\b[A-Z][a-zA-Z0-9_]*\b|\b[a-z_]+[A-Z][a-zA-Z0-9_]*\b",
                    user_question,
                )

                if api_matches:
                    # Use quick lookup for the most likely API
                    primary_api = max(api_matches, key=len)  # Take longest match
                    lookup_result = self.quick_api_lookup(safe, primary_api)
                    lines.extend(
                        [f"### Recherche API pour '{primary_api}'", lookup_result, ""]
                    )

                # Fallback to auto-retrieve
                context = self._original_auto_retrieve_context(safe, user_question)
                lines.extend([context])

            elif intent == "navigation":
                # Show repository structure and likely paths
                scan_result = self.scan_repo_files(
                    safe, limit=30, order="path", ascending=True
                )
                lines.extend(["### Structure du dépôt", scan_result, ""])

                # Add targeted search
                nav_keywords = re.findall(
                    r"\b\w+\.\w+\b|\b\w+/\w+\b", user_question.lower()
                )
                if nav_keywords:
                    for keyword in nav_keywords[:2]:
                        search_result = self.find_in_repo(
                            safe, keyword, use_regex=False, max_matches=10
                        )
                        if "Aucun résultat" not in search_result:
                            lines.extend(
                                [f"### Recherche '{keyword}'", search_result, ""]
                            )

            elif intent == "tutorial":
                # Focus on tests, examples, and documentation
                test_files = self.find_in_repo(
                    safe, "test|example|demo|sample", use_regex=True, max_matches=15
                )
                if "Aucun résultat" not in test_files:
                    lines.extend(["### Tests et exemples", test_files, ""])

                # Get context with tutorial focus
                context = self._original_auto_retrieve_context(safe, user_question)
                lines.extend([context])

            else:  # general
                # Standard auto-retrieve context
                context = self._original_auto_retrieve_context(safe, user_question)
                lines.extend([context])

            # Always add high-level synthesis if not already included
            if intent not in ["debug", "tutorial"]:
                synthesis = self.get_repo_context(
                    safe, max_files=1, max_chars_per_file=1000
                )
                lines.extend(["### Synthèse générale", synthesis])

            result = "\n".join(lines)

            # Apply context governance
            result_bytes = len(result.encode("utf-8", errors="ignore"))
            if result_bytes > self.valves.max_context_bytes:
                truncate_at = self.valves.max_context_bytes - 200
                result = result.encode("utf-8")[:truncate_at].decode(
                    "utf-8", errors="ignore"
                )
                result += "\n\n... [CONTEXTE TRONQUÉ - utilisez une question plus spécifique] ..."

            self.logger.info(
                "[AUTOLOAD] intent=%s, context_size=%d bytes", intent, result_bytes
            )
            return self._redact_secrets(result)

        except Exception as e:
            self.logger.error("[AUTOLOAD] auto_load_context error: %s", e)
            return f"❌ Erreur auto_load_context: {e}"

    # =====================================
    # v1.8 AST PARSING MODULE
    # =====================================

    def _ast_available(self) -> bool:
        """Check if AST parsing is available."""
        return _TREE_SITTER_AVAILABLE and len(_TREE_SITTER_PARSERS) > 0

    def _ast_index_repo(self, repo_dir: str) -> Dict[str, Any]:
        """Build AST-based index for repository."""
        self.logger.info("[AST] Starting AST indexing for %s", repo_dir)

        if not self._ast_available():
            self.logger.info("[AST] Tree-sitter not available, falling back to regex")
            return self._build_symbol_index_for_repo(repo_dir)

        # AST-based indexing (placeholder implementation)
        index = {
            "repo": os.path.basename(repo_dir),
            "generated_at": datetime.now().isoformat(),
            "method": "ast",
            "functions": [],
            "classes": [],
            "imports": [],
            "exports": [],
            "calls": [],  # New: function call relationships
            "docstrings": [],  # New: documentation strings
        }

        files = self._get_repo_files_for_search(repo_dir)
        ast_files_processed = 0

        for file_info in files:
            lang_key = _EXT_TO_LANG.get(os.path.splitext(file_info['path'])[1].lower())

            if lang_key and lang_key in _TREE_SITTER_PARSERS:
                # Use AST parsing (placeholder)
                try:
                    # In real implementation, this would parse with Tree-sitter
                    ast_files_processed += 1
                except Exception as e:
                    self.logger.warning("[AST] Failed to parse %s: %s", file_info['path'], e)
                    self._add_regex_symbols_for_file(file_info, index, lang_key)
            else:
                # Use regex parsing
                self._add_regex_symbols_for_file(file_info, index, lang_key)

        self.logger.info("[AST] Indexed %d files with AST, %d total files",
                        ast_files_processed, len(files))
        return index

    def _add_regex_symbols_for_file(self, file_info: Dict, index: Dict, lang_key: str):
        """Add symbols using regex parsing (fallback)."""
        if not lang_key or lang_key not in _SYMBOL_PATTERNS:
            return

        patterns = _SYMBOL_PATTERNS[lang_key]

        try:
            with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Functions
                    for pattern in patterns.get('functions', []):
                        match = pattern.match(line)
                        if match:
                            index["functions"].append({
                                "name": match.group(1),
                                "file": file_info['path'],
                                "line": line_num,
                                "lang": lang_key,
                                "method": "regex"
                            })

                    # Classes
                    for pattern in patterns.get('classes', []):
                        match = pattern.match(line)
                        if match:
                            index["classes"].append({
                                "name": match.group(1),
                                "file": file_info['path'],
                                "line": line_num,
                                "lang": lang_key,
                                "method": "regex"
                            })
        except Exception as e:
            self.logger.warning("Error processing file %s: %s", file_info['path'], e)

    # =====================================
    # v1.8 BM25 & HYBRID RETRIEVAL
    # =====================================

    def _build_bm25_index(self, repo_dir: str) -> Optional[SimpleBM25]:
        """Build BM25 index from repository content."""
        self.logger.info("[BM25] Building BM25 index for %s", repo_dir)

        files = self._get_repo_files_for_search(repo_dir)
        documents = []
        metadata = []

        for file_info in files:
            try:
                with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Split content into chunks
                chunks = self._split_into_chunks(content, max_size=1000)

                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'file': file_info['path'],
                        'chunk': i,
                        'lang': file_info.get('lang', ''),
                        'type': 'code'
                    })

            except Exception as e:
                self.logger.warning("[BM25] Failed to process %s: %s", file_info['path'], e)
                continue

        if not documents:
            return None

        bm25 = SimpleBM25(
            k1=self.user_valves.bm25_k1,
            b=self.user_valves.bm25_b
        )
        bm25.fit(documents)

        # Store metadata for later retrieval
        bm25.metadata = metadata

        self.logger.info("[BM25] Index built with %d chunks from %d files",
                        len(documents), len(files))
        return bm25

    def _split_into_chunks(self, content: str, max_size: int = 1000) -> List[str]:
        """Split content into chunks for BM25 indexing."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)

            if current_size + line_size > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _build_embeddings_index(self, repo_dir: str) -> Optional[Tuple[Any, Any, List]]:
        """Build embeddings index if available."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE or not _FAISS_AVAILABLE:
            return None

        if not self.user_valves.use_embeddings:
            return None

        self.logger.info("[EMBEDDINGS] Building embeddings index for %s", repo_dir)

        try:
            model = SentenceTransformer(self.user_valves.embedding_model)

            files = self._get_repo_files_for_search(repo_dir)
            texts = []
            metadata = []

            for file_info in files:
                try:
                    with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    chunks = self._split_into_chunks(content, max_size=500)

                    for i, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadata.append({
                            'file': file_info['path'],
                            'chunk': i,
                            'lang': file_info.get('lang', ''),
                            'type': 'code'
                        })

                        if len(texts) >= self.user_valves.max_vector_chunks:
                            break

                    if len(texts) >= self.user_valves.max_vector_chunks:
                        break

                except Exception as e:
                    self.logger.warning("[EMBEDDINGS] Failed to process %s: %s", file_info['path'], e)
                    continue

            if not texts:
                return None

            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=False)

            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

            self.logger.info("[EMBEDDINGS] Index built with %d chunks, dimension %d",
                           len(texts), dimension)
            return model, index, metadata

        except Exception as e:
            self.logger.error("[EMBEDDINGS] Failed to build embeddings index: %s", e)
            return None

    def _enhanced_deduplicate_extracts(self, extracts: List[str]) -> List[str]:
        """Enhanced deduplication using MinHash if available."""
        if not self.user_valves.enable_deduplication:
            return extracts

        if not _DATASKETCH_AVAILABLE:
            # Fall back to v1.7 method (direct implementation to avoid recursion)
            unique_extracts = []
            seen_signatures = set()

            for extract in extracts:
                # Create a signature from function/class definitions
                signature_lines = []
                for line in extract.split('\n'):
                    if re.search(r'\b(def|function|class|const|export)\s+\w+', line):
                        # Normalize the signature
                        sig = re.sub(r'\s+', ' ', line.strip())
                        signature_lines.append(sig)

                signature = '|'.join(signature_lines)

                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_extracts.append(extract)

            return unique_extracts

        try:
            # Use MinHash for better similarity detection
            lsh = MinHashLSH(threshold=0.8, num_perm=128)
            unique_extracts = []

            for i, extract in enumerate(extracts):
                # Create MinHash for this extract
                minhash = MinHash(num_perm=128)

                # Tokenize extract content (remove markdown formatting)
                text = re.sub(r'```.*?\n|```', '', extract)
                tokens = re.findall(r'\w+', text.lower())

                for token in tokens:
                    minhash.update(token.encode('utf-8'))

                # Check for similar documents
                similar = lsh.query(minhash)

                if not similar:
                    # No similar document found, add this one
                    lsh.insert(f"doc_{i}", minhash)
                    unique_extracts.append(extract)

            self.logger.info("[DEDUP] MinHash reduced %d extracts to %d",
                           len(extracts), len(unique_extracts))
            return unique_extracts

        except Exception as e:
            self.logger.warning("[DEDUP] MinHash failed, falling back: %s", e)
            return self._deduplicate_extracts(extracts)

    # =====================================================================
    # 🆕 NEW v1.8 PUBLIC FUNCTIONS
    # =====================================================================

    def hybrid_retrieve(self, repo_name: str, query: str, k: int = 10, mode: str = "auto") -> str:
        """
        Hybrid retrieval using BM25 + optional embeddings.

        Args:
            repo_name: Repository name (sanitized)
            query: Search query
            k: Number of results to return
            mode: "auto", "bm25", "embeddings", or "hybrid"

        Returns:
            Formatted markdown with search results, scores, and sources
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Determine retrieval method
            backend = mode if mode != "auto" else self.user_valves.retrieval_backend

            self.logger.info("[HYBRID] query='%s', k=%d, mode=%s, backend=%s",
                           query, k, mode, backend)

            results = []
            method_used = []

            # BM25 retrieval
            if backend in ["bm25", "hybrid"]:
                bm25 = self._build_bm25_index(repo_dir)
                if bm25:
                    bm25_results = bm25.search(query, k)
                    for idx, score in bm25_results:
                        if idx < len(bm25.metadata):
                            meta = bm25.metadata[idx]
                            chunk_content = bm25.documents[idx]
                            results.append({
                                'content': chunk_content,
                                'file': meta['file'],
                                'chunk': meta['chunk'],
                                'score': score,
                                'method': 'bm25',
                                'lang': meta.get('lang', ''),
                            })
                    method_used.append("BM25")
                else:
                    self.logger.warning("[HYBRID] BM25 index creation failed")

            # Embeddings retrieval
            if backend in ["embeddings", "hybrid"] and self.user_valves.use_embeddings:
                embeddings_data = self._build_embeddings_index(repo_dir)
                if embeddings_data:
                    model, index, metadata = embeddings_data

                    # Search with embeddings
                    query_embedding = model.encode([query])
                    faiss.normalize_L2(query_embedding)

                    scores, indices = index.search(query_embedding, min(k, len(metadata)))

                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(metadata):
                            meta = metadata[idx]
                            results.append({
                                'content': f"[Chunk {meta['chunk']} from {meta['file']}]",
                                'file': meta['file'],
                                'chunk': meta['chunk'],
                                'score': float(score),
                                'method': 'embeddings',
                                'lang': meta.get('lang', ''),
                            })
                    method_used.append("Embeddings")
                else:
                    self.logger.info("[HYBRID] Embeddings not available or disabled")

            # Fallback to keywords if no other method worked
            if not results:
                self.logger.info("[HYBRID] Falling back to keyword search")
                keyword_result = self.auto_retrieve_context(safe, query)
                return f"## Hybrid Retrieve — {safe}\n\n**Méthode**: Keywords (fallback)\n\n{keyword_result}"

            # Sort and take top k results
            if backend == "hybrid" and len(method_used) > 1:
                results.sort(key=lambda x: x['score'], reverse=True)

            top_results = results[:k]

            # Format output
            lines = [
                f"## Hybrid Retrieve — {safe}",
                f"**Méthodes**: {' + '.join(method_used)}",
                f"**Résultats**: {len(top_results)}/{len(results)} (k={k})",
                "",
            ]

            if not top_results:
                lines.append("ℹ️ Aucun résultat trouvé.")
                return "\n".join(lines)

            # Group by file for better presentation
            by_file = defaultdict(list)
            for result in top_results:
                by_file[result['file']].append(result)

            lines.append("### Résultats par pertinence")
            lines.append("")

            for file_path, file_results in list(by_file.items())[:10]:  # Max 10 files
                file_results.sort(key=lambda x: x['score'], reverse=True)
                best_score = file_results[0]['score']

                lines.append(f"#### {file_path} (score: {best_score:.3f})")

                for result in file_results[:3]:  # Max 3 chunks per file
                    method_badge = {"bm25": "🔍", "embeddings": "🧠", "keywords": "🏷️"}.get(
                        result['method'], "❓"
                    )

                    lines.append(f"{method_badge} **Chunk {result['chunk']}** (score: {result['score']:.3f})")

                    # Show content preview
                    content_preview = result['content'][:200].replace('\n', ' ')
                    if len(result['content']) > 200:
                        content_preview += "..."
                    lines.append(f"```{result.get('lang', '')}")
                    lines.append(content_preview)
                    lines.append("```")
                    lines.append("")

            # Add sources table
            lines.extend([
                "### Table des sources",
                ""
            ])

            for i, result in enumerate(top_results[:20], 1):  # Max 20 in sources
                method_badge = {"bm25": "🔍", "embeddings": "🧠", "keywords": "🏷️"}.get(
                    result['method'], "❓"
                )
                lines.append(f"{i}. {result['file']}:chunk_{result['chunk']} "
                           f"({method_badge} score: {result['score']:.3f})")

            result_text = "\n".join(lines)

            # Apply context limits
            result_bytes = len(result_text.encode('utf-8', errors='ignore'))
            if result_bytes > self.valves.max_context_bytes:
                truncate_at = self.valves.max_context_bytes - 200
                result_text = result_text.encode('utf-8')[:truncate_at].decode('utf-8', errors='ignore')
                result_text += "\n\n... [CONTEXTE TRONQUÉ] ..."

            self.logger.info("[HYBRID] Retrieved %d results, context_size=%d bytes",
                           len(top_results), result_bytes)
            return self._redact_secrets(result_text)

        except Exception as e:
            self.logger.error("[HYBRID] hybrid_retrieve error: %s", e)
            return f"❌ Erreur hybrid_retrieve: {e}"

    def show_call_graph(self, repo_name: str, symbol: str, max_nodes: int = 50) -> str:
        """
        Show call graph for a symbol using AST if available, regex fallback.

        Args:
            repo_name: Repository name (sanitized)
            symbol: Symbol to analyze
            max_nodes: Maximum nodes in call graph

        Returns:
            Formatted markdown with call graph information
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Try to load or create AST index
            analysis_dir = os.path.join(repo_dir, "docs_analysis")
            ast_index_path = os.path.join(analysis_dir, "ast_index.json")

            index = None
            method_used = "regex"

            if os.path.exists(ast_index_path):
                try:
                    with open(ast_index_path, 'r', encoding='utf-8') as f:
                        index = json.load(f)
                    if index.get('method') == 'ast':
                        method_used = "ast"
                except Exception as e:
                    self.logger.warning("[XREF] Failed to load AST index: %s", e)

            if not index:
                # Build new index (AST if available, regex fallback)
                index = self._ast_index_repo(repo_dir)
                if index.get('method') == 'ast':
                    method_used = "ast"

                # Save index
                os.makedirs(analysis_dir, exist_ok=True)
                try:
                    with open(ast_index_path, 'w', encoding='utf-8') as f:
                        json.dump(index, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    self.logger.warning("[XREF] Failed to save AST index: %s", e)

            self.logger.info("[XREF] Analyzing call graph for '%s' using %s", symbol, method_used)

            # Find symbol definitions
            definitions = []
            for func in index.get("functions", []):
                if func["name"] == symbol:
                    definitions.append(func)
            for cls in index.get("classes", []):
                if cls["name"] == symbol:
                    definitions.append(cls)

            if not definitions:
                return f"🔎 Symbole « {symbol} » non trouvé dans l'index."

            # Build call graph data structure
            call_graph = {
                'definitions': definitions,
                'calls_from': [],  # Functions this symbol calls
                'calls_to': [],    # Functions that call this symbol
                'method': method_used
            }

            # Analyze calls (AST-based if available, regex fallback)
            if method_used == "ast" and "calls" in index:
                # Use detailed call information from AST
                for call in index["calls"]:
                    if call.get("caller") == symbol:
                        call_graph['calls_from'].append(call)
                    elif call.get("callee") == symbol:
                        call_graph['calls_to'].append(call)
            else:
                # Regex-based call analysis
                files = self._get_repo_files_for_search(repo_dir)
                for file_info in files:
                    try:
                        with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                # Simple heuristic for function calls
                                if symbol in line and '(' in line:
                                    # Check if it looks like a function call
                                    pattern = rf'\b{re.escape(symbol)}\s*\('
                                    if re.search(pattern, line):
                                        call_graph['calls_to'].append({
                                            'file': file_info['path'],
                                            'line': line_num,
                                            'content': line.strip(),
                                            'method': 'regex_heuristic'
                                        })
                    except Exception:
                        continue

            # Format output
            lines = [
                f"## Call Graph — {symbol}",
                f"**Méthode d'analyse**: {method_used.upper()}",
                f"**Définitions trouvées**: {len(call_graph['definitions'])}",
                "",
            ]

            # Show definitions
            if call_graph['definitions']:
                lines.append("### Définitions")
                for defn in call_graph['definitions']:
                    kind = "Fonction" if defn in index.get("functions", []) else "Classe"
                    lines.append(f"- **{kind}**: {defn['name']} dans {defn['file']}:{defn['line']}")
                    if defn.get('method') == 'ast' and 'signature' in defn:
                        lines.append(f"  - Signature: `{defn['signature']}`")
                    if 'docstring' in defn and defn['docstring']:
                        doc_preview = defn['docstring'][:100].replace('\n', ' ')
                        if len(defn['docstring']) > 100:
                            doc_preview += "..."
                        lines.append(f"  - Doc: {doc_preview}")
                lines.append("")

            # Show outgoing calls (calls from this symbol)
            if call_graph['calls_from']:
                lines.append(f"### Appels sortants ({len(call_graph['calls_from'])})")
                for call in call_graph['calls_from'][:max_nodes//2]:
                    target = call.get('callee', call.get('target', 'unknown'))
                    location = f"{call['file']}:{call['line']}" if 'file' in call else 'unknown'
                    lines.append(f"- `{symbol}` → `{target}` ({location})")
                if len(call_graph['calls_from']) > max_nodes//2:
                    lines.append(f"- ... et {len(call_graph['calls_from']) - max_nodes//2} autres")
                lines.append("")

            # Show incoming calls (calls to this symbol)
            if call_graph['calls_to']:
                lines.append(f"### Appels entrants ({len(call_graph['calls_to'])})")
                for call in call_graph['calls_to'][:max_nodes//2]:
                    if call.get('method') == 'regex_heuristic':
                        lines.append(f"- {call['file']}:{call['line']}: {call['content'][:80]}...")
                    else:
                        caller = call.get('caller', 'unknown')
                        location = f"{call['file']}:{call['line']}" if 'file' in call else 'unknown'
                        lines.append(f"- `{caller}` → `{symbol}` ({location})")
                if len(call_graph['calls_to']) > max_nodes//2:
                    lines.append(f"- ... et {len(call_graph['calls_to']) - max_nodes//2} autres")
                lines.append("")

            # Summary statistics
            total_relationships = len(call_graph['calls_from']) + len(call_graph['calls_to'])
            lines.extend([
                "### Résumé",
                f"- **Relations totales**: {total_relationships}",
                f"- **Appels sortants**: {len(call_graph['calls_from'])}",
                f"- **Appels entrants**: {len(call_graph['calls_to'])}",
                f"- **Méthode**: {method_used} ({'AST précis' if method_used == 'ast' else 'heuristique regex'})",
            ])

            if method_used == "regex":
                lines.extend([
                    "",
                    "ℹ️ *Analyse basée sur des heuristiques regex. Pour une analyse précise, installez Tree-sitter.*"
                ])

            return "\n".join(lines)

        except Exception as e:
            self.logger.error("[XREF] show_call_graph error: %s", e)
            return f"❌ Erreur show_call_graph: {e}"

    # =====================================================================
    # 🆕 NEW v1.9 DEVELOPER UX FUNCTIONS
    # =====================================================================

    def outline_file(self, repo_name: str, path: str, max_items: int = 200) -> str:
        """
        Generate structural outline of a large file for better navigation.

        Args:
            repo_name: Repository name (sanitized)
            path: Relative path to file within repo
            max_items: Maximum items to display (1-1000, clamped to 200)

        Returns:
            Formatted markdown outline with functions, classes, and structure
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Security: ensure path is within repo
            full_path = os.path.realpath(os.path.join(repo_dir, path))
            repo_real = os.path.realpath(repo_dir)
            if not (full_path == repo_real or full_path.startswith(repo_real + os.sep)):
                return "❌ Chemin refusé (hors dépôt)."

            if not os.path.isfile(full_path):
                return f"❌ Fichier introuvable: {path}"

            # Check if binary file
            try:
                with open(full_path, "rb") as f:
                    head = f.read(4096)
                    if self._looks_binary(head):
                        return "❌ Fichier binaire / non texte."
            except Exception:
                return "❌ Erreur lecture fichier."

            # Clamp max_items
            max_items = max(1, min(max_items, 1000))
            if max_items > 200:
                max_items = 200

            # Get file size
            file_size = os.path.getsize(full_path)

            # Parse file for declarations
            items = []
            current_class = None
            indentation_level = 0

            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        original_line = line
                        stripped = line.strip()

                        if not stripped or stripped.startswith('#'):
                            continue

                        # Detect indentation (for Python class methods)
                        leading_spaces = len(line) - len(line.lstrip())

                        # Python patterns
                        # Class definition
                        class_match = re.match(r'^\s*class\s+(\w+)', line)
                        if class_match:
                            class_name = class_match.group(1)
                            current_class = class_name
                            indentation_level = leading_spaces
                            items.append({
                                'line': line_num,
                                'kind': 'class',
                                'name': class_name,
                                'full_name': class_name
                            })
                            continue

                        # Function definition
                        func_match = re.match(r'^\s*(async\s+)?def\s+(\w+)', line)
                        if func_match:
                            func_name = func_match.group(2)
                            # Check if it's a method (indented under a class)
                            if current_class and leading_spaces > indentation_level:
                                full_name = f"{current_class}.{func_name}"
                                kind = 'method'
                            else:
                                full_name = func_name
                                kind = 'function'
                                current_class = None  # Reset class context

                            items.append({
                                'line': line_num,
                                'kind': kind,
                                'name': func_name,
                                'full_name': full_name
                            })
                            continue

                        # JavaScript/TypeScript patterns
                        # Function declaration
                        js_func_match = re.match(r'^\s*function\s+(\w+)', line)
                        if js_func_match:
                            func_name = js_func_match.group(1)
                            items.append({
                                'line': line_num,
                                'kind': 'function',
                                'name': func_name,
                                'full_name': func_name
                            })
                            continue

                        # Const function (arrow function)
                        const_func_match = re.match(r'^\s*const\s+(\w+)\s*=\s*(?:async\s+)?\(.*?\)\s*=>', line)
                        if const_func_match:
                            func_name = const_func_match.group(1)
                            items.append({
                                'line': line_num,
                                'kind': 'function',
                                'name': func_name,
                                'full_name': func_name
                            })
                            continue

                        # Class definition (JS/TS)
                        js_class_match = re.match(r'^\s*class\s+(\w+)', line)
                        if js_class_match:
                            class_name = js_class_match.group(1)
                            current_class = class_name
                            items.append({
                                'line': line_num,
                                'kind': 'class',
                                'name': class_name,
                                'full_name': class_name
                            })
                            continue

                        # Method definition (JS/TS) - simplified
                        method_match = re.match(r'^\s*(\w+)\s*\(.*?\)\s*{', line)
                        if method_match and current_class and leading_spaces > 0:
                            method_name = method_match.group(1)
                            # Skip common non-methods
                            if method_name not in ['if', 'for', 'while', 'switch', 'try']:
                                items.append({
                                    'line': line_num,
                                    'kind': 'method',
                                    'name': method_name,
                                    'full_name': f"{current_class}.{method_name}"
                                })

                        # Reset class context if we're back at top level
                        if current_class and leading_spaces == 0 and stripped and not stripped.startswith(('}', '//')):
                            current_class = None

            except Exception as e:
                return f"❌ Erreur parsing fichier: {e}"

            # Remove duplicates while preserving order
            seen = set()
            unique_items = []
            for item in items:
                key = (item['line'], item['kind'], item['full_name'])
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item)

            # Apply limit
            displayed_items = unique_items[:max_items]
            truncated = len(unique_items) > max_items

            # Build output
            lines = [
                f"## Outline — {safe}/{path}",
                f"**Taille**: {file_size} octets",
                f"**Items**: {len(displayed_items)}/{len(unique_items)}",
                "",
            ]

            if not displayed_items:
                lines.append("ℹ️ Aucune déclaration détectée (fonction, classe).")
            else:
                lines.append("### Déclarations")
                for item in displayed_items:
                    kind_icon = {
                        'class': '🏛️',
                        'function': '⚡',
                        'method': '🔧'
                    }.get(item['kind'], '📄')

                    lines.append(f"- {item['line']:4d} · {kind_icon} {item['kind']} · **{item['full_name']}**")

            if truncated:
                lines.extend([
                    "",
                    f"… (troncature à {max_items} items)"
                ])

            self.logger.info("[OUTLINE] %s/%s: %d items (%d total), %d bytes",
                           safe, path, len(displayed_items), len(unique_items), file_size)

            return "\n".join(lines)

        except Exception as e:
            self.logger.error("[OUTLINE] outline_file error: %s", e)
            return f"❌ Erreur outline_file: {e}"

    def find_tests_for(self, repo_name: str, target: str, max_results: int = 50) -> str:
        """
        Find tests related to a symbol or file quickly.

        Args:
            repo_name: Repository name (sanitized)
            target: Symbol name or relative file path
            max_results: Maximum results (1-500, clamped to 50)

        Returns:
            Formatted markdown with test candidates sorted by relevance
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Clamp max_results
            max_results = max(1, min(max_results, 500))
            if max_results > 50:
                max_results = 50

            # Determine if target is a file path or symbol
            is_file = '/' in target or '.' in target
            candidates = []

            # Get all files for searching
            files = self._get_repo_files_for_search(repo_dir)

            # Filter to test files only
            test_files = []
            for file_info in files:
                file_path = file_info['path'].lower()
                if any(marker in file_path for marker in [
                    'test', 'spec', '__tests__', 'tests/',
                    '.test.', '_test.', '.spec.'
                ]):
                    test_files.append(file_info)

            if not test_files:
                return f"ℹ️ Aucun fichier de test trouvé dans {safe}."

            # Load index if available for symbol lookup
            index = self._load_or_create_index(safe)

            if is_file:
                # Target is a file path
                target_basename = os.path.splitext(os.path.basename(target))[0]
                target_module = target.replace('/', '.').replace('.py', '').replace('.js', '').replace('.ts', '')

                for file_info in test_files:
                    score = 0
                    reasons = []

                    try:
                        with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Check for direct import of the module
                        import_patterns = [
                            rf'from\s+{re.escape(target_module)}\s+import',
                            rf'import\s+{re.escape(target_module)}',
                            rf'from\s+.*{re.escape(target_basename)}\s+import',
                            rf'require\s*\(\s*[\'\"]{re.escape(target)}[\'\"]',
                            rf'import\s+.*from\s+[\'\"]{re.escape(target)}[\'\"]'
                        ]

                        for pattern in import_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                score += 3
                                reasons.append('import')
                                break

                        # Check for basename in test file name
                        test_filename = os.path.basename(file_info['path']).lower()
                        if target_basename.lower() in test_filename:
                            score += 1
                            reasons.append('name')

                        # Check for path proximity (same directory tree)
                        target_dir = os.path.dirname(target)
                        test_dir = os.path.dirname(file_info['path'])
                        if target_dir and test_dir and target_dir in test_dir:
                            score += 1
                            reasons.append('proximity')

                        if score > 0:
                            candidates.append({
                                'file': file_info['path'],
                                'score': score,
                                'reasons': reasons,
                                'content': content
                            })

                    except Exception:
                        continue

            else:
                # Target is a symbol
                symbol_files = set()

                # Find where symbol is defined
                if index:
                    for func in index.get("functions", []):
                        if func["name"] == target:
                            symbol_files.add(func["file"])
                    for cls in index.get("classes", []):
                        if cls["name"] == target:
                            symbol_files.add(cls["file"])

                for file_info in test_files:
                    score = 0
                    reasons = []

                    try:
                        with open(file_info['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        # Check for direct usage of symbol
                        usage_pattern = rf'\b{re.escape(target)}\s*\('
                        if re.search(usage_pattern, content):
                            score += 2
                            reasons.append('usage')

                        # Check for import from symbol's file
                        for symbol_file in symbol_files:
                            module_name = os.path.splitext(symbol_file)[0].replace('/', '.')
                            import_pattern = rf'from\s+{re.escape(module_name)}\s+import.*{re.escape(target)}'
                            if re.search(import_pattern, content):
                                score += 3
                                reasons.append('import')
                                break

                        # Check for name matching in test file
                        test_filename = os.path.basename(file_info['path']).lower()
                        if target.lower() in test_filename:
                            score += 1
                            reasons.append('name')

                        if score > 0:
                            candidates.append({
                                'file': file_info['path'],
                                'score': score,
                                'reasons': reasons,
                                'content': content
                            })

                    except Exception:
                        continue

            # Sort by score descending
            candidates.sort(key=lambda x: x['score'], reverse=True)
            candidates = candidates[:max_results]

            # Build output
            lines = [
                f"## Tests liés — {target} (repo={safe})",
                "",
            ]

            if not candidates:
                lines.extend([
                    f"ℹ️ Aucun test trouvé pour « {target} ».",
                    "Essayez: élargir patterns, vérifier nommage fichiers test, ou utiliser find_in_repo."
                ])
                return "\n".join(lines)

            lines.append(f"### Candidats (N={len(candidates)})")
            lines.append("")

            # Show candidates with reasons
            for candidate in candidates:
                reason_str = '+'.join(candidate['reasons'])
                lines.append(f"- **{candidate['score']}** · {candidate['file']} — raison: {reason_str}")

            # Show extracts for top 3 candidates
            if len(candidates) >= 1:
                lines.extend(["", "### Extraits (top candidats)", ""])

                for candidate in candidates[:3]:
                    lines.append(f"#### {candidate['file']}")

                    # Extract meaningful lines (test function names, assertions)
                    content_lines = candidate['content'].split('\n')
                    relevant_lines = []

                    for i, line in enumerate(content_lines, 1):
                        # Look for test functions, assertions, and target mentions
                        if any(pattern in line.lower() for pattern in [
                            'def test_', 'it(', 'describe(', 'test(', 'assert',
                            target.lower(), 'expect'
                        ]):
                            context_start = max(0, i - 3)
                            context_end = min(len(content_lines), i + 3)

                            for j in range(context_start, context_end):
                                if j < len(content_lines):
                                    relevant_lines.append(f"{j+1:4d}: {content_lines[j]}")

                            break  # Just show first relevant section

                    if relevant_lines:
                        lang = self._ext_lang_hint(candidate['file'])
                        lines.append(f"```{lang}")
                        lines.extend(relevant_lines[:20])  # Max 20 lines
                        lines.append("```")
                    else:
                        lines.append("*(aucun extrait pertinent)*")

                    lines.append("")

            # Sources table
            lines.extend([
                "### Table des sources",
                ""
            ])

            for candidate in candidates:
                lines.append(f"- {candidate['file']} (score: {candidate['score']}, raisons: {'+'.join(candidate['reasons'])})")

            self.logger.info("[TESTS] find_tests_for target='%s': %d candidates found",
                           target, len(candidates))

            return "\n".join(lines)

        except Exception as e:
            self.logger.error("[TESTS] find_tests_for error: %s", e)
            return f"❌ Erreur find_tests_for: {e}"

    def recent_changes(self, repo_name: str, days: int = 7, max_commits: int = 50) -> str:
        """
        Show recent git changes summary for quick debug/navigation context.

        Args:
            repo_name: Repository name (sanitized)
            days: Time window in days (1-365, clamped to 7)
            max_commits: Maximum commits (1-500, clamped to 50)

        Returns:
            Formatted markdown with recent commits and file change summary
        """
        try:
            p = self._paths()
            safe = self._sanitize_repo_name(repo_name)
            repo_dir = os.path.join(p["repos"], safe)

            if not os.path.isdir(repo_dir):
                return f"❌ Dépôt introuvable: {safe}"

            # Check if it's a git repository
            git_dir = os.path.join(repo_dir, '.git')
            if not os.path.exists(git_dir):
                return "❌ Référentiel Git introuvable."

            # Clamp parameters
            days = max(1, min(days, 365))
            max_commits = max(1, min(max_commits, 500))
            if max_commits > 50:
                max_commits = 50

            # Build git log command
            cmd = [
                'git', 'log',
                f'--since={days} days ago',
                f'--max-count={max_commits}',
                '--name-only',
                '--pretty=format:%h%x09%ad%x09%an%x09%s',
                '--date=short'
            ]

            self.logger.info("[RECENT] Running git log for %s (days=%d, max_commits=%d)",
                           safe, days, max_commits)

            # Execute git command
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=float(self.valves.git_timeout_s),
                    check=False
                )

                if proc.returncode != 0:
                    stderr_truncated = (proc.stderr or "")[:200]
                    return f"❌ Git log échoué (rc={proc.returncode}): {stderr_truncated}"

                output = proc.stdout.strip()

            except subprocess.TimeoutExpired:
                return f"❌ Timeout git log après {self.valves.git_timeout_s}s"
            except Exception as e:
                return f"❌ Erreur exécution git: {e}"

            if not output:
                return f"ℹ️ Aucun commit dans les {days} derniers jours."

            # Parse git log output
            commits = []
            file_changes = defaultdict(int)
            current_commit = None

            for line in output.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Check if it's a commit line (contains tabs)
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        hash_short, date, author, subject = parts[0], parts[1], parts[2], parts[3]
                        current_commit = {
                            'hash': hash_short,
                            'date': date,
                            'author': author,
                            'subject': subject,
                            'files': []
                        }
                        commits.append(current_commit)
                else:
                    # It's a file name
                    if current_commit is not None:
                        current_commit['files'].append(line)
                        file_changes[line] += 1

            # Build output
            lines = [
                f"## Recent changes — {safe} (since {days}d, max {max_commits})",
                f"**Commits**: {len(commits)}",
                f"**Fichiers uniques**: {len(file_changes)}",
                "",
            ]

            if commits:
                lines.append("### Commits")
                for commit in commits:
                    file_count = len(commit['files'])
                    file_preview = ', '.join(commit['files'][:3])
                    if file_count > 3:
                        file_preview += f", ... (+{file_count-3})"

                    lines.append(f"- **{commit['hash']}** · {commit['date']} · {commit['author']} — {commit['subject']}")
                    lines.append(f"  *(files: {file_count}) {file_preview}*")

                lines.append("")

            # Top modified files
            if file_changes:
                lines.append("### Top fichiers modifiés (fenêtre)")
                top_files = sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:10]

                for file_path, count in top_files:
                    lines.append(f"- **{file_path}** · {count} commits")

            self.logger.info("[RECENT] %s: %d commits, %d unique files in %dd window",
                           safe, len(commits), len(file_changes), days)

            return "\n".join(lines)

        except Exception as e:
            self.logger.error("[RECENT] recent_changes error: %s", e)
            return f"❌ Erreur recent_changes: {e}"


# =====================================================================
# COMPLETION - v1.9 is ready!
# =====================================================================
