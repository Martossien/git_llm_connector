# 📐 Implementation Plan R2 — Git LLM Connector

*Généré automatiquement le 2025-09-12 20:22 (Europe/Paris)*

## 1. Résumé exécutif
R2 vise à transformer le Git LLM Connector en un outil robuste pour les mono‐repos massifs tout en restant 100 % CLI et sans dépendances lourdes. Les nouveautés clés portent sur la scalabilité (clone shallow/sparse), l’analyse incrémentale, un « Context Builder » repensé avec chunking+scoring, un RAG local optionnel, des valves additionnelles et une redaction renforcée. L’objectif est de livrer un lot prêt à release, documenté, testé et compatible Open WebUI ≥ 0.6.0.

## 2. Architecture actuelle
```
./git_llm_connector.py
├── Valves & UserValves
├── _run_git_command / _resolve_executable
├── _scan_repository_files
├── _run_llm_analysis
└── docs/
    └── RAPPORT_IMPLEMENTATION.md
```
R2 étend cette architecture en ajoutant de nouvelles fonctions privées (_sparse_checkout_init, _rag_build_or_update_index…), un modèle de métadonnées enrichi et des tests couvrant toutes les stories.

## 3. Spécifications détaillées
### EPIC A — Scalabilité Git
**Valves admin**
```json
{
  "git_clone_depth": 0,
  "git_sparse_patterns": ""
}
```
- `git_clone_depth>0` ⇒ `git clone --depth N`
- `git_sparse_patterns` ⇒ comma‐separated patterns (gitignore syntax)

**Fonctions**
```python
async def _sparse_checkout_init(repo_dir: str) -> None
async def _sparse_checkout_set(repo_dir: str, patterns: List[str]) -> None
```
Pseudo‑code pour `_sparse_checkout_set`:
1. `await _run_git_command(repo_dir,["sparse-checkout","set",*patterns])`
2. Log des chemins inclus via event emitter

### EPIC B — Analyse incrémentale / diff
**Valves user**
```json
{
  "analysis_mode": "full"  # ou "diff"
}
```
**Métadonnées** `docs_analysis/analysis_metadata.json`
```json
{
  "repo_head_commit": "<sha>",
  "generated_at": "2025-09-12T20:22:00Z",
  "files": [
    {"path": "src/app.py", "size": 1234, "sha256": "…", "analyzed_at": "2025-09-12T20:20:00Z"}
  ],
  "llm_config": {...},
  "scan_config": {...}
}
```
**Fonctions**
```python
async def _compute_file_sha256(path: Path) -> str
async def _load_previous_metadata(meta_path: Path) -> Optional[Dict]
```
Diff‑mode : comparer SHA256, ne sélectionner que fichiers nouveaux/altérés. Fallback full si metadata absente ou corrompue.

### EPIC C — Context Builder 2.0
**Valves admin**
```json
{
  "chunk_max_bytes": 16384,
  "chunk_overlap_bytes": 512
}
```
**Fonctions**
```python
async def _chunk_file(path: Path, max_bytes: int, overlap: int) -> List[Chunk]
async def _score_chunks(chunks: List[Chunk]) -> List[Chunk]
async def _build_context_budgeted(chunks: List[Chunk], max_context_bytes: int) -> Tuple[str,List[Chunk]]
```
**Scoring** (ordre décroissant)
1. Fichiers prioritaires: `README*`, manifests, config build
2. Proximité racine ou dossiers `src/`, `app/`, `server/`
3. Extensions code (`.py`, `.ts`, …) vs docs/tests
4. Taille moyenne favorisée
Déduplication par hash SHA256 du chunk.

Contexte final accompagné d’une section “Excluded due to budget” listant les 10 chunks suivants.

### EPIC D — RAG local optionnel
**Valves admin**
```json
{
  "rag_enabled": false,
  "rag_top_k": 8
}
```
**Valves user**
```json
{
  "rag_query_boost": 0.0
}
```
**Index** : répertoire `rag_index/`
- Tokenisation: `re.split(r"[^a-z0-9]+", text.lower())`
- Stopwords intégrées (`{"the","and","or","la","le"}`…)
- Inverted index `{token: [(chunk_id, freq), ...]}`
- BM25 light :
  \(
  \text{score}(q,d) = \sum_{t\in q} IDF(t) \cdot \frac{f(t,d) (k+1)}{f(t,d) + k(1-b + b\,|d|/\text{avgdl})}
  \)
  avec `k=1.5`, `b=0.75`, `IDF(t)=log(1 + (N - n_t + 0.5)/(n_t + 0.5))`

**Fonctions**
```python
async def _rag_build_or_update_index(repo_dir: Path, chunks: List[Chunk]) -> None
async def _rag_retrieve(repo_dir: Path, query: str, top_k: int) -> List[RagHit]
```
Si `analysis_mode="diff"`, indexer uniquement les nouveaux chunks.

### EPIC E — UX, valves & logs
Nouveaux événements UI :
- `📦 Sparse checkout…`
- `🔍 Scan repo…`
- `🧩 Build context 60%…`
- `📚 RAG query…`
- `📝 Synthèse 2/3…`

### EPIC F — Sécurité & robustesse
- Redaction patterns supplémentaires: `OPENAI_API_KEY`, `GCP_*_KEY`, `AZURE_*_KEY`, blocs PEM `-----BEGIN PRIVATE KEY-----`
- Détection binaire : ratio `\x00` > 0.2 ⇒ skip
- `errors="ignore"` sur tous `open()`, timeouts + `proc.kill()` sur dépassement

### EPIC G — Tests & CI
**Unit tests**
- `test_sparse_checkout.py`
- `test_diff_mode.py`
- `test_chunking.py`
- `test_rag.py`
- `test_redaction_plus.py`
**Intégration** : repo git local mocké, LLM simulé par `echo`.
**CI** : `.github/workflows/tests.yml` → matrice `{3.11,3.12}`.

## 4. Modèle de données `analysis_metadata.json`
Exemple complet :
```json
{
  "repo_head_commit": "abc123",
  "generated_at": "2025-09-12T20:22:00Z",
  "files": [
    {
      "path": "src/main.py",
      "size": 2048,
      "sha256": "deadbeef…",
      "analyzed_at": "2025-09-12T20:00:00Z"
    }
  ],
  "llm_config": {
    "llm_bin_name": "qwen",
    "llm_model_name": "qwen2.5-coder-32b-instruct"
  },
  "scan_config": {
    "include": ["src/**"],
    "exclude": ["tests/**"],
    "max_file_size_kb": 256
  }
}
```

## 5. Algorithme BM25 light
- Calcul `IDF` pré‐stocké par token
- Stockage des longueurs de document `|d|`
- Recherche : tokeniser la requête, accumuler scores, trier
- Renvoie liste `[{chunk_id, path, score}]`

## 6. Stratégies de test
### Commandes
```bash
pytest -q                      # unitaires
pytest -q tests/integration    # intégration
```
Couverture attendue :
- Sparse checkout limite bien le scan
- Diff mode ne renvoie que fichiers modifiés
- Chunking respecte overlap et budget
- RAG retourne documents pertinents
- Redaction masque nouvelles clés

## 7. Limites & pistes
- Pas d’embeddings ni de similarité avancée (option future)
- Index BM25 naïf non persistant entre machines
- Pas de gestion MCP/agents multiples
- Performance dépendante d’I/O disque

## 8. Sécurité & risques
- Absence totale de `shell=True`
- Validation stricte des templates LLM via `shlex.split`
- Fichiers binaires ignorés pour éviter injection
- Timeouts obligatoires sur tous sous‑processus

## 9. Checklist release
- [ ] Implémentation complète des EPIC A–F
- [ ] Ajout tests EPIC G
- [ ] CI Python 3.11/3.12 verte
- [ ] README mis à jour (valves, exemples CLI)
- [ ] Tag version + publication

---
*Fin du document — prêt pour exécution des stories R2.*
