"""
title: Git LLM Connector
author: Martossien
author_url: https://github.com/Martossien
git_url: https://github.com/Martossien/git_llm_connector
description: v1.6 safe — Lecture + Git + Analyse LLM via CLI (gemini/qwen). Contexte borné, logs détaillés, chemins stables, LLM timeout=900s.
required_open_webui_version: 0.6.0
version: 0.1.6
license: MIT
requirements: aiofiles,pathspec,pydantic
"""

from typing import Any, List, Dict
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

# -------------------------------------
# UTILITAIRE LOCAL (petite redaction)
# -------------------------------------
_SECRET_PATTERNS = [
    (re.compile(r"(?i)(API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*[^\s]+"), r"\1=****"),
    (re.compile(r"ghp_[A-Za-z0-9]+"), "ghp_****"),
    (re.compile(r"eyJ[\w-]+?\.[\w-]+?\.[\w-]+"), "****"),  # JWT grossier
]


class Tools:
    """
    Git LLM Connector — v1.6 SAFE

    ✅ Public (signatures simples):
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
    """

    # ------------------------------
    # Valves (admin)
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
    # UserValves (utilisateur)
    # ------------------------------
    class UserValves(BaseModel):
        llm_cli_choice: str = Field(
            default="qwen", description="LLM CLI (qwen / gemini / auto)."
        )
        analysis_mode: str = Field(
            default="smart", description="Mode d’analyse (réservé v2)."
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

    # ------------------------------
    # Init
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

        # Forcer chemins et timeout
        self.valves.git_repos_path = self.repos_dir
        self.valves.llm_timeout_s = 900.0

        self._setup_logging()
        self.logger.info(
            "[INIT] v1.6 safe — base=%s, repos=%s, logs=%s, llm_timeout=%ss",
            self.base_dir,
            self.repos_dir,
            self.logs_dir,
            self.valves.llm_timeout_s,
        )

    # ------------------------------
    # Logging
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
    # Helpers
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

    # -------------------------------------
    # LLM helpers (CLI)
    # -------------------------------------
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

    # -------------------------------------
    # Contexte de code
    # -------------------------------------
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

        # fichiers prioritaires
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

        # 1) priority
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

        # 2) scan fichiers eligibles
        depth = (depth or self.user_valves.analysis_depth or "standard").lower()
        limits = {"quick": 10, "standard": 25, "deep": 50}
        max_files = limits.get(depth, 25)

        files: List[str] = []
        for root, dirs, filenames in os.walk(repo_dir):
            rel_root = os.path.relpath(root, repo_dir)
            if rel_root == ".":
                rel_root = ""

            # filtres dossiers exclus
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

        # on garde les max_files premiers (ordre OS)
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
    # 🔓 FONCTIONS PUBLIQUES (lecture + git manuel + LLM)
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
                f"- Affiché: jusqu’à {max_bytes} octets",
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

    # ------------------------------
    # NEW — clean_analysis
    # ------------------------------
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

    # ------------------------------
    # NEW — llm_check
    # ------------------------------
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

    # ------------------------------
    # NEW — analyze_repo (LLM)
    # ------------------------------
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
                "tool_version": "1.6.0",
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

