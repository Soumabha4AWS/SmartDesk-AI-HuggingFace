# Smart Desk AI — README

A walk-through of each component and how to run the project.

## 1) Overview
Smart Desk AI is a LangGraph-based multi-agent system for HR/IT support with Retrieval-Augmented Generation (RAG), ticket status lookup, and ticket escalation. It supports multi-conversation (threaded) memory.

## 2) Prerequisites
- Windows 10/11
- Python 3.12+
- Internet access on first start to download the local chat and embedding models

## 3) Setup
```powershell
# Create and activate venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create .env from the example only if it does not already exist
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Optionally set HF_TOKEN to a Hugging Face access token for authenticated downloads.
```

## 4) Project Structure
```text
project/
  agents/
    kb_agent.py               # HR/IT RAG agent
    ticket_creation_agent.py  # Escalation agent
    ticket_status_agent.py    # Ticket status by ID/email; sets last_ticket_id
  documents/                  # KB sources used by RAG
    hr-policies.md, it-support.md, onboarding.md, payroll.md
  models/                     # Downloaded local GGUF model (excluded from Git)
  ragSetup/
    rag_setup.py              # Build embeddings, vector store, retriever
  tickets/
    synthetic_tickets.json    # Mock ticket dataset (shown as source "tickets.db")
  smart_desk_ai_agent.py      # Graph assembly, state, routing, checkpointer, ask()

run_project/ui_smart_desk_ai_agent.py  # Gradio ChatInterface
run_project/test_smart_desk_ai_agent.py # Script harness with (query, thread_id) tuples
QuickStart.md                   # One-page setup
Architecture.md                 # Diagrams and data flow
```

## 5) RAG Pipeline (project/ragSetup/rag_setup.py)
- Loads markdown documents.
- Splits with RecursiveCharacterTextSplitter.
- Embeds locally via HuggingFaceEmbeddings using
  `sentence-transformers/all-MiniLM-L6-v2`, then stores vectors in Chroma.
- Exposes `retriever` (k configurable; MMR optional) and `format_docs`.

## 6) Local Chat Model
- Chat inference uses `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` through
  `llama-cpp-python`; it does not send chat prompts to the AI Router.
- If `HF_TOKEN` is set in the repository-root `.env`, it authenticates Hugging
  Face model downloads. It is optional for these public models.
- On first startup, the Q4_K_M GGUF file is downloaded to
  `project/models/TinyLlama-1.1B-Chat-v1.0-GGUF/`. The model file is excluded
  from Git. Later starts reuse the downloaded file.

## 7) Core Agent Graph (project/smart_desk_ai_agent.py)
- SupportState fields:
  - `query`, `intent`, `response`, `context`, `retrieved_sources`, `history`, `effective_query`, `last_ticket_id`.
- Nodes in LangGraph:
  - `rewrite_query` — resolves anaphora (e.g., “that ticket”) using `last_ticket_id`.
  - `classify_intent` — routes to `kb_query`, `tkct_status`, or `tkct_creation`.
  - `kb_query_agent` — RAG over documents (sources are the markdown filenames).
  - `ticket_status_agent` — lookup by ID/email; sets `last_ticket_id`; sources shown as `tickets.db`.
  - `ticket_creation_agent` — collects details and provides escalation channels.
- Checkpointer:
  - Default: `MemorySaver` (in-process).
  - Optional: `SqliteSaver` for durable memory (install `langgraph-checkpoint-sqlite`).
- `ask(app, query, thread_id)`:
  - Only sends `{ "query": ... }` so persisted state is preserved per `thread_id`.

## 8) UI (run_project/ui_smart_desk_ai_agent.py)
- Gradio ChatInterface with a conversation selector and a New conversation button.
- Each conversation has an independent LangGraph thread and keeps its chat transcript while switching between conversations in the current browser session.
- Exact questions are cached per conversation for the lifetime of the app process, so repeating one skips the agent/LLM call; matching ignores case and repeated whitespace.
- Displays the detected intent and retrieved sources for each response.

## 9) Test Harness (test_smart_desk_ai_agent.py)
- `TEST_QUERIES` supports either a string or a `(query, thread_id)` tuple.
- Use the same `thread_id` to demonstrate multi-conversation memory and query-rewrite:
```python
# python
TEST_QUERIES = {
  "Q1 — Thread B": ("Give me the status for Ticket created for marry.smith@gmail.com", "cust-B"),
  "Q2 — Thread B": ("Give me the description and priority for that Ticket", "cust-B"),
}
```

## 10) Retrieval Tuning
- Build with higher recall and reranking:
```python
# python
agent = build_support_agent(top_k=3, enable_reranking=True, rerank_fetch_k=6)
```
- Optional MMR in `rag_setup.py`:
```python
# python
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "lambda_mult": 0.4})
```

## 11) Run
```powershell
# Chat UI
python .\run_project\ui_smart_desk_ai_agent.py

# Script
python .\run_project\test_smart_desk_ai_agent.py
```

## 12) Troubleshooting
- Ensure `llama-cpp-python` is installed and allow first startup to download the
  GGUF and embedding models.
- Use `from langgraph.checkpoint.memory import MemorySaver`.
- Continue in the same selected conversation for follow-ups.
- If documents change, consider rebuilding the vector store (new `collection_name`).

---
