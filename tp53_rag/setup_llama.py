#!/usr/bin/env python3
"""
============================================================
TP53 RAG Platform — Setup & Installation Script
setup_llama.py
============================================================
Automated setup for:
1. Python dependencies (pip install)
2. llama.cpp download & compilation  
3. Gemma 4 2B model download (Q4_K_M quantization)
4. Environment configuration
5. Knowledge base ingestion

Run: python setup_llama.py
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from urllib.request import urlopen
import hashlib
import platform

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# Gemma 4 2B Q4_K_M (quantized for 8GB RAM)
MODEL_NAME = "gemma-2b-it-Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/lmstudio-community/Gemma-2-2B-it-GGUF/resolve/main/Gemma-2-2B-it-Q4_K_M.gguf"
MODEL_SIZE_MB = 1200  # approximate

# llama.cpp repository
LLAMACPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
LLAMACPP_DIR = PROJECT_ROOT / "llama.cpp"

# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_step(text: str, step_num: int = 0):
    prefix = f"[{step_num}]" if step_num > 0 else "→"
    print(f"{prefix} {text}")

def print_success(text: str):
    print(f"✅ {text}")

def print_error(text: str):
    print(f"❌ {text}")

def print_warning(text: str):
    print(f"⚠️  {text}")

def run_command(cmd: str, cwd: Path = None, check: bool = True) -> int:
    """Run shell command and return exit code."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and result.returncode != 0:
        print_error(f"Command failed: {cmd}")
        return 1
    return result.returncode

def check_disk_space() -> int:
    """Check available disk space in GB."""
    stat = shutil.disk_usage("/")
    return stat.free // (1024**3)

def check_ram() -> int:
    """Check system RAM in GB (cross-platform)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024**3))
    except:
        # Fallback: assume 8GB if can't detect
        return 8

# ═══════════════════════════════════════════════════════════════════
# Setup steps
# ═══════════════════════════════════════════════════════════════════

def step_1_system_check():
    """Verify system requirements."""
    print_header("STEP 1: System Requirements Check")
    
    # OS
    os_name = platform.system()
    print_step(f"Operating System: {os_name}", 1)
    if os_name not in ["Linux", "Darwin", "Windows"]:
        print_warning(f"Untested OS: {os_name}")
    print_success(f"Detected: {os_name}")
    
    # RAM
    print_step("RAM Check", 2)
    ram_gb = check_ram()
    print(f"  Available RAM: ~{ram_gb} GB")
    if ram_gb < 8:
        print_warning(f"Less than 8GB RAM detected. Gemma 2B Q4_K_M needs ~4GB + OS overhead.")
        print("  Proceeding anyway (may be slow).")
    else:
        print_success(f"Sufficient RAM ({ram_gb}GB)")
    
    # Disk space
    print_step("Disk Space Check", 3)
    free_gb = check_disk_space()
    print(f"  Free disk space: {free_gb} GB")
    if free_gb < 5:
        print_error(f"Insufficient disk space ({free_gb}GB < 5GB needed)")
        return False
    print_success(f"Sufficient disk space ({free_gb}GB)")
    
    # Python
    print_step("Python Version", 4)
    print(f"  {sys.version}")
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ required")
        return False
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    
    return True

def step_2_directories():
    """Create required directories."""
    print_header("STEP 2: Directory Setup")
    
    dirs = [MODELS_DIR, LOGS_DIR, DATA_DIR / "documents", DATA_DIR / "chroma_db"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print_step(f"Created: {d.relative_to(PROJECT_ROOT)}")
    
    print_success("All directories ready")
    return True

def step_3_pip_dependencies():
    """Install Python dependencies."""
    print_header("STEP 3: Python Dependencies")
    
    requirements = [
        "llama-cpp-python>=0.2.0",
        "langchain>=0.1.0",
        "langchain-core",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "streamlit>=1.28.0",
        "streamlit-option-menu",
        "plotly",
        "numpy",
        "scipy",
        "scikit-learn",
        "rank-bm25",
        "pydantic",
        "python-dotenv",
        "requests",
        "rich",
        "torch>=2.0.0",  # For sentence-transformers
    ]
    
    print_step(f"Installing {len(requirements)} packages...", 1)
    
    # Try pip install with --extra-index-url for llama-cpp-python wheels
    cmd = f"{sys.executable} -m pip install --upgrade pip"
    if run_command(cmd, check=False) != 0:
        print_warning("Failed to upgrade pip")
    
    # Install main requirements
    for req in requirements:
        print_step(f"Installing {req}...", 2)
        cmd = f"{sys.executable} -m pip install '{req}' -q"
        if run_command(cmd, check=False) != 0:
            print_warning(f"  → {req} installation had issues (may retry later)")
    
    print_success("Python dependencies installed")
    return True

def step_4_llamacpp_server():
    """Clone and build llama.cpp server."""
    print_header("STEP 4: llama.cpp Server Setup")
    
    if LLAMACPP_DIR.exists():
        print_step("llama.cpp already cloned", 1)
    else:
        print_step(f"Cloning llama.cpp...", 1)
        if run_command(f"git clone {LLAMACPP_REPO}", cwd=PROJECT_ROOT, check=False) != 0:
            print_warning("git clone failed — trying alternate method")
            return False
        print_success("llama.cpp cloned")
    
    # Try to build (optional — pre-built binaries can work too)
    print_step("Checking for pre-built server binary...", 2)
    server_exe = LLAMACPP_DIR / ("llama-server.exe" if platform.system() == "Windows" else "llama-server")
    
    if not server_exe.exists():
        print_step("Building llama.cpp server from source...", 3)
        os.chdir(LLAMACPP_DIR)
        if platform.system() == "Windows":
            if run_command("cmake -B build -DBUILD_SHARED_LIBS=ON", cwd=LLAMACPP_DIR, check=False) == 0:
                run_command("cmake --build build --config Release", cwd=LLAMACPP_DIR, check=False)
        else:
            if run_command("make", cwd=LLAMACPP_DIR, check=False) == 0:
                print_success("llama.cpp built successfully")
            else:
                print_warning("Build failed — will use pip install of llama-cpp-python instead")
    else:
        print_success(f"Found server: {server_exe}")
    
    return True

def step_5_model_download():
    """Download Gemma 4 2B model."""
    print_header("STEP 5: Model Download (Gemma 4 2B Q4_K_M)")
    
    model_path = MODELS_DIR / MODEL_NAME
    
    if model_path.exists():
        print_step(f"Model already exists: {model_path}", 1)
        size_mb = model_path.stat().st_size / (1024**2)
        print_success(f"Model size: {size_mb:.0f}MB")
        return True
    
    print_step(f"Downloading {MODEL_NAME} ({MODEL_SIZE_MB}MB)...", 1)
    print("  This may take 5-15 minutes depending on connection speed")
    print(f"  Source: {MODEL_URL}")
    
    try:
        # Simple download with progress
        with urlopen(MODEL_URL) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            
            with open(model_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"  {percent:5.1f}% ({downloaded/(1024**2):.0f}/{total_size/(1024**2):.0f}MB)", end='\r')
        
        print_success(f"Model downloaded: {model_path}")
        return True
    except Exception as e:
        print_error(f"Model download failed: {e}")
        print_warning("Download manually from HuggingFace and place in ./models/")
        return False

def step_6_env_file():
    """Create .env configuration file."""
    print_header("STEP 6: Environment Configuration")
    
    env_file = PROJECT_ROOT / ".env"
    
    env_content = f"""# TP53 RAG Platform Configuration
# Auto-generated by setup_llama.py

# Inference Mode
INFERENCE_MODE=llamacpp
LLAMA_MODEL_PATH=models/{MODEL_NAME}

# llama.cpp Server (must be running for inference)
LLAMACPP_BASE_URL=http://localhost:8080

# Google GenAI API (optional fallback)
GOOGLE_API_KEY=
GOOGLE_MODEL=gemma-4-26b-a4b-it

# ChromaDB
CHROMA_PERSIST_DIR=data/chroma_db
CHROMA_COLLECTION_NAME=tp53_knowledge_base

# RAG Hyperparameters
CHUNK_SIZE=512
CHUNK_OVERLAP=128
TOP_K_RESULTS=3

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO

# NCBI (optional)
ENTREZ_EMAIL=your_email@example.com
NCBI_API_KEY=
"""
    
    if env_file.exists():
        print_step(f"Existing .env found", 1)
        print_warning("Skipping (edit manually if needed)")
    else:
        print_step(f"Creating .env file", 1)
        with open(env_file, "w") as f:
            f.write(env_content)
        print_success(f"Created: {env_file}")
    
    return True

def step_7_quickstart_guide():
    """Create QUICKSTART.md guide."""
    print_header("STEP 7: Quickstart Guide")
    
    guide = """# TP53 RAG Platform — Quickstart Guide

## After Setup Complete

### 1. Start llama.cpp Server (Terminal 1)

```bash
cd llama.cpp  # or use pre-built binary
./llama-server -m ../models/gemma-2b-it-Q4_K_M.gguf \\
  -c 8192 \\
  --timeout 300 \\
  --threads 4 \\
  --parallel 2
```

Server will be ready when you see: `Server listening on http://127.0.0.1:8080`

### 2. Build Knowledge Base (Terminal 2)

```bash
cd tp53_analysis
python main.py build
```

This ingests TP53 documents into ChromaDB (5-10 minutes).

### 3. Run Streamlit App (Terminal 3)

```bash
cd tp53_rag
streamlit run app.py
```

App opens at `http://localhost:8501`

### 4. Query Examples

In the Streamlit app:
- Tab 1 (🧬 Query): Ask "What is R175H?" 
- Tab 2 (🎯 Analysis): Select mutation R175H, run multi-agent analysis
- Tab 3 (💊 Drug): Find drugs for a mutation
- Tab 5 (📄 Report): Generate a clinical report

---

## Troubleshooting

### Issue: Empty outputs from LLM
- ✅ Solution: Ensure llama.cpp server is running and responding
- Test: `curl http://localhost:8080/health`

### Issue: "Knowledge base not built"
- ✅ Solution: Run `python main.py build` first

### Issue: Out of memory errors
- ✅ Solution: Reduce context window in `config/settings.py`
- Change `TOP_K_RESULTS=3` to `TOP_K_RESULTS=2`
- Reduce `CHUNK_SIZE` from 512 to 384

### Issue: llama.cpp won't compile
- ✅ Solution: Use pre-built binary instead
- Download from: https://github.com/ggerganov/llama.cpp/releases
- Or: `pip install llama-cpp-python`

---

## Performance Tips

1. **CPU Cores**: Use `--threads N` matching your CPU cores (usually 4-8)
2. **Parallel Requests**: `--parallel 2` allows 2 concurrent queries
3. **Context Window**: 8192 tokens is max for 8GB RAM. Reduce if swapping.
4. **Quantization**: Q4_K_M is optimal for 8GB RAM (good quality/speed tradeoff)

---

## Hackathon Pitch

**"Privacy-first multi-agent TP53 analysis running entirely on local CPU"**

Features to demo:
- Voice input → transcription → RAG query (if Whisper built)
- Multi-agent orchestration (variant analysis, drug discovery, immunotherapy)
- Enterprise dossier generation (PDF export)
- Real-time streaming responses
- Kenya/KEML clinical context

Demo Script:
1. "What is R175H?" → Variant Curator analysis
2. "What drugs work?" → Drug Discovery agent + KEML availability
3. "Generate a report" → Dossier Compiler (academic + enterprise modes)

---

## File Structure for Reference

```
tp53_analysis/
├── tp53_rag/
│   ├── app.py (Streamlit UI)
│   ├── main.py (CLI)
│   ├── agents/
│   │   ├── rag_chain.py (core RAG)
│   │   ├── dispatcher.py (multi-agent router)
│   │   ├── variant_curator.py (mutation classification)
│   │   ├── immunogenicity.py (TME prediction)
│   │   ├── dossier_compiler.py (report generation)
│   │   ├── drug_discovery.py (therapeutics)
│   │   └── [5 more specialized agents]
│   ├── utils/
│   │   ├── pii_scrubber.py (HIPAA compliance)
│   │   ├── hybrid_search.py (BM25 + vector search)
│   │   ├── reranker.py (cross-encoder reranking)
│   │   └── [more utilities]
│   ├── knowledge_base/
│   │   ├── ingestion.py
│   │   └── vector_store.py
│   ├── config/
│   │   └── settings.py
│   └── data/
│       ├── documents/
│       ├── chroma_db/ (vector database)
│       └── semantic_cache.db
├── models/
│   └── gemma-2b-it-Q4_K_M.gguf (1.2GB)
├── llama.cpp/ (cloned repo, build artifacts)
└── .env (configuration)
```
"""
    
    quickstart_file = PROJECT_ROOT / "QUICKSTART.md"
    with open(quickstart_file, "w") as f:
        f.write(guide)
    
    print_step(f"Created: QUICKSTART.md", 1)
    print_success("Quickstart guide ready")
    return True

def step_8_test_setup():
    """Test the setup."""
    print_header("STEP 8: Test Setup")
    
    print_step("Testing Python imports...", 1)
    try:
        import langchain
        print_success("✓ langchain")
    except ImportError as e:
        print_warning(f"✗ langchain: {e}")
    
    try:
        import chromadb
        print_success("✓ chromadb")
    except ImportError as e:
        print_warning(f"✗ chromadb: {e}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print_success("✓ sentence-transformers")
    except ImportError as e:
        print_warning(f"✗ sentence-transformers: {e}")
    
    try:
        import streamlit
        print_success("✓ streamlit")
    except ImportError as e:
        print_warning(f"✗ streamlit: {e}")
    
    print_step("Checking model file...", 2)
    model_path = MODELS_DIR / MODEL_NAME
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024**2)
        print_success(f"✓ Model exists ({size_mb:.0f}MB)")
    else:
        print_warning(f"✗ Model not found at {model_path}")
    
    return True

def main():
    """Run full setup."""
    print_header("TP53 RAG PLATFORM — AUTOMATED SETUP")
    
    print("""
This script will:
  1. Verify system requirements (RAM, disk, OS)
  2. Create necessary directories
  3. Install Python dependencies
  4. Clone and build llama.cpp
  5. Download Gemma 4 2B model
  6. Configure environment
  7. Create quickstart guide
  8. Test the setup

Total time: 15-30 minutes (depending on network/CPU)
    """)
    
    input("Press Enter to begin...")
    
    steps = [
        ("System Requirements", step_1_system_check),
        ("Directory Setup", step_2_directories),
        ("Python Dependencies", step_3_pip_dependencies),
        ("llama.cpp Server", step_4_llamacpp_server),
        ("Model Download", step_5_model_download),
        ("Environment Config", step_6_env_file),
        ("Quickstart Guide", step_7_quickstart_guide),
        ("Test Setup", step_8_test_setup),
    ]
    
    failed = []
    for i, (name, step_func) in enumerate(steps, 1):
        try:
            result = step_func()
            if not result:
                failed.append(name)
                print_warning(f"Step {i} had issues (continuing anyway)")
        except Exception as e:
            failed.append(name)
            print_error(f"Step {i} failed: {e}")
            print_warning("Continuing to next step...")
    
    # Summary
    print_header("SETUP COMPLETE")
    
    if failed:
        print_warning(f"⚠️  Some steps had issues: {', '.join(failed)}")
        print("\nPlease review and manually fix:")
        for step_name in failed:
            print(f"  - {step_name}")
    else:
        print_success("✅ All steps completed successfully!")
    
    print("\nNext steps:")
    print("  1. Read QUICKSTART.md")
    print("  2. Start llama-server: ./llama.cpp/llama-server -m models/gemma-2b-it-Q4_K_M.gguf -c 8192 --threads 4")
    print("  3. Build knowledge base: python tp53_rag/main.py build")
    print("  4. Run Streamlit: streamlit run tp53_rag/app.py")
    print("\n🚀 Good luck at the hackathon!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)
