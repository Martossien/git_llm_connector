# Git LLM Connector v1.9 - Experimental Tool for Open WebUI

**Status: Experimental** - Local Git repository analysis using LLM CLI tools.

An experimental tool that combines Git cloning with local LLM analysis to explore code repositories privately. Think of it as a basic alternative to online code analysis services, but everything runs on your machine.

## What It Does

- **Clone Git repositories** locally for analysis
- **Generate AI summaries** using free LLM CLI tools (Qwen, Gemini)  
- **Search and explore code** with various strategies
- **Extract context** for AI conversations about your code
- **Navigate repositories** with specialized functions

**Key benefit**: Your code never leaves your machine - all analysis is done locally.

## Installation

### Prerequisites

**Node.js 20+** (required for LLM CLI tools):
```bash
node -v  # Check if installed
# If not: install via nvm or your system's package manager
```

**Git** (configured):
```bash
git --version
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### LLM CLI Setup

Choose one:

**Qwen CLI (Free, recommended)**:
```bash
npm install -g @qwen-code/qwen-code
qwen  # First run: authenticate with qwen.ai account
```

**Gemini CLI (Free tier)**:
```bash
npm install -g @google/gemini-cli  
gemini  # First run: authenticate with Google account
```

### Open WebUI Installation

1. Go to **Workspace** → **Tools** → **+**
2. Delete default template code
3. Paste the entire `git_llm_connector.py` content
4. Save
5. Activate in conversations using **+**

## Available Functions (18 total)

### Basic Operations
- `tool_health()` - Check system status
- `list_repos()` - List cloned repositories
- `git_clone(url, name)` - Clone a repository
- `git_update(repo_name)` - Update repository

### Repository Analysis  
- `analyze_repo(repo_name, sections, depth)` - Generate AI analysis (5-30 min)
- `get_repo_context(repo_name)` - Load pre-generated summaries
- `repo_info(repo_name)` - Repository metadata

### Code Exploration
- `scan_repo_files(repo_name, limit, order)` - File inventory
- `preview_file(repo_name, path)` - View file content  
- `find_in_repo(repo_name, pattern, use_regex)` - Search in code
- `outline_file(repo_name, path)` - File structure (functions, classes)

### Context & Search
- `auto_retrieve_context(repo_name, question)` - Smart context extraction
- `hybrid_retrieve(repo_name, query, k, mode)` - Advanced search
- `quick_api_lookup(repo_name, symbol)` - Find symbol definitions
- `find_usage_examples(repo_name, symbol)` - Usage patterns
- `show_call_graph(repo_name, symbol)` - Function relationships

### Developer Tools
- `find_tests_for(repo_name, target)` - Locate related tests
- `recent_changes(repo_name, days)` - Git history summary
- `show_related_code(repo_name, path)` - File dependencies

## Basic Workflow

1. **Clone a repository**:
```python
git_clone("https://github.com/user/project")
```

2. **Generate AI analysis** (takes time):
```python
analyze_repo("user_project", "architecture,api", "quick")  # 5-15 minutes
```

3. **Ask questions with context**:
```python
get_repo_context("user_project")
auto_retrieve_context("user_project", "how does authentication work?")
```

## Context System

The tool builds context for AI conversations through:

1. **High-level summaries** (from LLM analysis):
   - `ARCHITECTURE.md` - System overview
   - `API_SUMMARY.md` - Key functions and classes
   - `CODE_MAP.md` - Navigation guide

2. **Dynamic retrieval** (from search):
   - Keyword matching in code
   - BM25 text search (optional)
   - Code excerpts with context windows

3. **Specialized searches**:
   - Symbol definitions and usage
   - Test file discovery
   - Cross-references and dependencies

## Performance Reality

| Operation | Time | Notes |
|-----------|------|--------|
| Git clone | 30s-5min | Depends on repo size |
| LLM analysis | 5-30min | Can be slow for large repos |
| File search | <1s | Fast |
| Context retrieval | 1-5s | Depends on repo size |

**Memory usage**: 50MB-1GB depending on repository size and features used.

## Configuration

Basic settings through Open WebUI valves:

- `llm_cli_choice`: "qwen" or "gemini" or "auto"
- `analysis_depth`: "quick", "standard", or "deep"  
- `retrieval_backend`: "keywords", "bm25", or "hybrid"
- `max_context_bytes`: Context size limit (default 32MB)

## Limitations & Known Issues

- **Large repositories**: Analysis can be very slow (30+ minutes)
- **Memory usage**: Can consume significant RAM with large codebases
- **LLM dependencies**: Requires external authentication with Qwen/Gemini
- **Binary files**: Skipped automatically
- **Language support**: Best with Python/JavaScript/TypeScript, basic for others

## Troubleshooting

**LLM CLI not working**:
```bash
qwen --version  # or gemini --version
# If failed: check Node.js version, re-authenticate
```

**Analysis timeouts**:
- Use `depth="quick"` for initial exploration
- Increase timeout in admin settings
- Try smaller repositories first

**Out of memory**:
- Reduce `max_context_bytes` 
- Use `retrieval_backend="keywords"` (lighter than BM25/embeddings)
- Filter out large directories in settings

**Repository not found**:
```python
list_repos()  # Check exact name
```

## Storage Structure

```
~/git_llm_connector/
├── git_repos/           # Cloned repositories
│   └── {repo_name}/
│       ├── .git/        # Full git repository
│       ├── docs_analysis/  # Generated summaries
│       │   ├── ARCHITECTURE.md
│       │   ├── API_SUMMARY.md  
│       │   └── CODE_MAP.md
│       └── [source files]
└── logs/                # Operation logs
```

## Security Notes

- All processing happens locally
- Source code is not sent to external services  
- Only analysis prompts (not code) sent to LLM APIs
- Automatic redaction of API keys/secrets in logs
- Git credentials use your existing local configuration

## License

MIT License - experimental software, use at your own risk.

## Notes

- This is experimental software - expect bugs and limitations
- Performance varies significantly with repository size
- LLM analysis quality depends on the chosen model
- Best used for exploring unfamiliar codebases, not production workflows
- Consider it a local alternative to online code analysis tools

For detailed logs and debugging, check `~/git_llm_connector/logs/`.
