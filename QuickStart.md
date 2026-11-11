# Smart Desk AI — Quick Start

This guide shows how to set up and run the Smart Desk AI app (UI and script) on Windows.

## 1) Prerequisites
- Windows 10/11
- Python 3.12+ (`py -V`)
- Internet access on first start to download the local chat and embedding models

## 2) Create and activate a virtual environment
```powershell
# From the repository root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3) Install dependencies
```powershell
pip install -r requirements.txt
# Optional (durable thread memory via SQLite)
pip install -U langgraph-checkpoint-sqlite
```

## 4) Configure environment
```powershell
# Create .env from the example only if it does not already exist
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Optionally set HF_TOKEN to a Hugging Face access token for authenticated downloads.
```
Embeddings run locally via
`sentence-transformers/all-MiniLM-L6-v2`. Chat inference runs locally with
TinyLlama via `llama-cpp-python`. If set, `HF_TOKEN` from `.env` authenticates
Hugging Face model downloads; it is optional for these public models.

## 5) Local chat model
- The first startup downloads `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (Q4_K_M)
  to `project/models/TinyLlama-1.1B-Chat-v1.0-GGUF/`.
- The downloaded GGUF is reused on subsequent starts and is not tracked by Git.

## 6) Run the UI (Gradio)
```powershell
python .\run_project\ui_smart_desk_ai_agent.py
```
- The ChatInterface starts a conversation automatically. Use **New conversation** to create another independent thread, and use the conversation selector to switch between threads.
- Conversation transcripts and LangGraph memory are retained per thread while the app process and browser session are active.
- Repeating the same question in one conversation reuses its cached answer during the current app run; the in-memory cache clears when the app restarts.
- Examples:
  - "Give me the status for Ticket created for marry.smith@gmail.com"
  - then, in the same conversation: "Give me the description and priority for that Ticket"

## 7) Run the script tests
```powershell
python .\run_project\test_smart_desk_ai_agent.py
```
- The test harness supports tuples `(query, thread_id)` so you can group related questions under the same thread.

## 8) Retrieval quality options (optional)
You can increase recall and improve ranking when asking KB questions like "Code of conduct":
- In both UI and tests, build the agent with:
```python
# python
agent = build_support_agent(
    collection_name="smartdesk_demo",
    top_k=3,
    enable_reranking=True,
    rerank_fetch_k=6,
)
```
- (Alternative) Use MMR in `project/ragSetup/rag_setup.py`:
```python
# python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": top_k, "lambda_mult": 0.4},
)
```

## 9) Multi‑conversation memory
- The app persists per‑thread state using LangGraph’s checkpointer.
- Default: in‑memory (`MemorySaver`) — resets on process restart.
- Durable (optional): enable SQLite in `project/smart_desk_ai_agent.py` and install `langgraph-checkpoint-sqlite`.

```python
# python
# from langgraph.checkpoint.sqlite import SqliteSaver
# db_path = Path(__file__).parent / "data" / "conversations.db"
# db_path.parent.mkdir(parents=True, exist_ok=True)
# saver = SqliteSaver(str(db_path))
# app = graph.compile(checkpointer=saver)
```

## 10) Troubleshooting
- Hugging Face download authentication errors: verify `HF_TOKEN` in `.env`; it is optional for the public models used here.
- Model download or inference errors: ensure `huggingface_hub`, `sentence-transformers`, and `llama-cpp-python` are installed and that `project/models/` is writable.
- ImportError for MemorySaver: use `from langgraph.checkpoint.memory import MemorySaver`.
- Follow‑up not remembering the ticket: ensure the same Thread ID is set and you’re not reinitializing state (the included `ask()` only sends the new `query`).
- Sources show tickets.db for ticket answers from the synthetic dataset; KB answers show the specific Markdown file.

## 11) Project entry points
- UI: [ui_smart_desk_ai_agent.py](./ui_smart_desk_ai_agent.py)
- Script: [test_smart_desk_ai_agent.py](./test_smart_desk_ai_agent.py)
- Core agent: [project/smart_desk_ai_agent.py](./project/smart_desk_ai_agent.py)
