"""
Smart Desi AI Agent : Multi-Agent Smart Desk AI System
------------------------------------------------------------
Shared module that builds the multi-agent support system for IT & HR Operations .

Architecture:
  Query → Supervisor (intent classifier)
        → RAG Agent (RAG over KB Query)
        → Create Ticket Agent (mock ticket creation)
        → Ticket Status Agent (mock ticket status)
"""

# ── Standard library imports ──
import sys
import os
import logging
import re       # Used to extract account IDs (e.g. ACC-12345) from user queries
import unicodedata  # Unicode normalization for robust ID extraction
import json     # Used to serialize mock account data into LLM-readable context
from pathlib import Path
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Chroma phones home to PostHog by default. Its bundled client is also incompatible
# with current posthog releases and logs "Failed to send telemetry event" on every
# run, so switch telemetry off and silence that logger.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# ── LangChain / LangGraph imports ──
# RecursiveCharacterTextSplitter: splits HR&IT docs into overlapping chunks for embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Document: lightweight wrapper holding page_content + metadata (source filename)
from langchain_core.documents import Document
# Chroma: in-process vector store used for semantic retrieval (RAG)
from langchain_chroma import Chroma
# ChatPromptTemplate: builds system + human message pairs for each agent
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser: extracts the raw string content from an LLM response message
from langchain_core.output_parsers import StrOutputParser
# RunnablePassthrough: passes the input unchanged — used in the RAG chain to
#   forward the user question while the retriever branch fetches context
from langchain_core.runnables import RunnablePassthrough
# StateGraph: declarative graph builder for multi-agent orchestration
# END: sentinel node that terminates graph execution
# START: sentinel node that initiates graph execution
from langgraph.graph import StateGraph, END, START
# Split agents into separate modules (refactor-ready)
from agents.kb_agent import make_kb_agent
from agents.ticket_status_agent import make_ticket_status_agent
from agents.ticket_creation_agent import make_ticket_creation_agent

# Absolute path to the /project/documents/ folder containing the four hr&it operation
# markdown files that the KB Agent uses for Retrieval-Augmented Generation.
DOCUMENTS_DIR = Path(__file__).parent / "documents"
MODEL_REPOSITORY = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_DOWNLOAD_DIR = Path(__file__).parent / "models" / "TinyLlama-1.1B-Chat-v1.0-GGUF"

# ── Mock ticket database ──
# In production this would be a real database call.  
MOCK_TICKETS = {
    "TICK-001": {
            "ticket_id": "001",
            "title": "Users unable to log in after password reset",
            "description": "Multiple users reporting authentication failures after performing password reset. Error message: 'Invalid credentials'. Issue started after recent security patch deployment.",
            "resolution": "Found that the password hash algorithm was updated but session tokens weren't invalidated. Solution: Clear all active sessions and force re-authentication. Implemented automatic session cleanup on password change.",
            "category": "Authentication",
            "priority": "High",
            "created_date": "2024-01-15",
            "resolved_date": "2024-01-15"
    },
    "TICK-002": {
            "ticket_id": "002",
            "title": "Database connection timeout in production",
            "description": "Application experiencing intermittent 500 errors. Logs show 'connection pool exhausted' and 'timeout waiting for connection'. Affects approximately 20% of requests during peak hours.",
            "resolution": "Database connection pool was sized too small for peak load. Increased max_connections from 20 to 100. Added connection pooling monitoring and alerts. Optimized long-running queries that were holding connections.",
            "category": "Database",
            "priority": "Critical",
            "created_date": "2024-01-18",
            "resolved_date": "2024-01-18"
    },
    "TICK-003": {
            "ticket_id": "003",
            "title": "Payment processing fails for international cards",
            "description": "Customers using non-US credit cards receiving 'Payment declined' errors. Payment gateway returns error code 402. Issue only affects cards issued outside United States.",
            "resolution": "Payment gateway configuration was missing international card processing flag. Updated gateway settings to enable international transactions. Added country-specific currency conversion. Retested with cards from EU, UK, and APAC regions.",
            "category": "Payment",
            "priority": "High",
            "created_date": "2024-01-20",
            "resolved_date": "2024-01-21"
    },
}

# ── LangGraph shared state ──
# Every node in the graph reads from and writes to this typed dictionary.
# LangGraph merges each node's returned dict back into this state automatically.
class SupportState(TypedDict):
    query: str                    # The original customer question
    intent: str                   # Classified intent: "kb_query" | "tkct_creation" | "tkct_status"
    response: str                 # Final answer returned to the customer
    context: str                  # Retrieved context (hr&it chunks or account JSON)
    retrieved_sources: list[str]  # Source filenames of retrieved docs (for traceability)
    history: list[dict[str, str]]  # Conversation history: [{"role":"user","content":...}, {"role":"assistant","content":...}]
    effective_query: str          # Rewritten query with resolved references (if any)
    last_ticket_id: str           # Canonical last-resolved ticket id (e.g., TICK-002)

# Default system prompt for the HR&IT Operation Agent.  Key design choices:
#   - "based ONLY on the provided HR documents" → prevents hallucination
#   - Explicit fallback phrasing → gives a graceful "I don't know" answer
#   - PII guardrail instruction → defense-in-depth even before Module C adds
#     programmatic guardrails
# Callers can override this via the `hr_n_it_system_prompt` parameter in
# build_support_agent() to test different prompting strategies .
DEFAULT_HR_N_IT_SYSTEM_PROMPT = (
    "You are a helpful customer support agent for HR and IT Operations.\n"
    "Answer the customer's question based ONLY on the provided documents.\n"
    "If the answer is not found in the provided context, say:\n"
    "\"I'm sorry, I don't have information about that in our current query. "
    "Please contact our support team at support@securebank.com for further assistance.\"\n"
    "Do not make up information. Be concise, friendly, and professional.\n"
    "NEVER disclose sensitive account data (SSN, full account numbers) in your responses."
)


def build_support_agent(
    collection_name="support_docs_multi",
    chunk_size=1000,
    chunk_overlap=100,
    top_k=3,
    model=MODEL_REPOSITORY,
    hr_n_it_system_prompt=None,
    enable_reranking=False,
    rerank_fetch_k=None,
):
    """
    Build and return the multi-agent FinTech support system.

    This is the main factory function.  It:
      1. Loads the four hr-&-it markdown files from disk
      2. Chunks them with RecursiveCharacterTextSplitter
      3. Embeds chunks into an in-memory Chroma vector store
      4. Wires up three specialist agents + a supervisor into a LangGraph

    Args:
        collection_name : Chroma collection name (change to avoid collisions
                          when running multiple tests in the same process).
        chunk_size      : Maximum characters per chunk.
        chunk_overlap   : Overlap between consecutive chunks (helps preserve
                          context at chunk boundaries).
        top_k           : Number of chunks the retriever returns per query.
        model           : Hugging Face GGUF repository for local chat inference.
        hr_n_it_system_prompt : Override the default HR&IT Agent system prompt                       .
        enable_reranking: If True, fetch more docs and re-sort by relevance
                          score before trimming to top_k (Module B exercise).
        rerank_fetch_k  : How many docs to fetch before reranking
                          (defaults to top_k * 2 when enable_reranking=True).

    Returns:
        dict with keys:
            app         : Compiled LangGraph application (invoke with ask())
            retriever   : The Chroma vector store retriever
            format_docs : Helper function that formats docs into a string
            llm         : The local TinyLlama chat model instance
            rag_chain   : The standalone HR&IT RAG chain (retriever → prompt → LLM)
            vectorstore : The underlying Chroma vector store
    """
    # ─── RAG pipeline (docs → chunks → embeddings → Chroma → retriever) ───
    from ragSetup.rag_setup import build_rag_pipeline
    rag = build_rag_pipeline(
        documents_dir=DOCUMENTS_DIR,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )
    vectorstore = rag["vectorstore"]
    retriever = rag["retriever"]
    format_docs = rag["format_docs"]

    # Download the quantized GGUF once into project/models, then run inference locally.
    from huggingface_hub import hf_hub_download
    from langchain_community.chat_models import ChatLlamaCpp

    MODEL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    model_path = hf_hub_download(
        repo_id=model,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DOWNLOAD_DIR,
        token=hf_token,
    )
    llm = ChatLlamaCpp(
        model_path=model_path,
        temperature=0,
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False,
    )

    # ─── Step 4: Build the HR-IT Agent RAG chain ───
    # This standalone chain is also exposed in the returned dict so Module B can
    # invoke it directly for evaluation without going through the full graph.
    hr_n_it_sys_prmpt = hr_n_it_system_prompt or DEFAULT_HR_N_IT_SYSTEM_PROMPT
    hr_n_it_prompt = ChatPromptTemplate.from_messages([
        ("system", DEFAULT_HR_N_IT_SYSTEM_PROMPT),
        ("human",
         "Context from our hr&it documents:\n\n{context}\n\n"
         "Customer question: {question}"),
    ])
    # The RAG chain works in two parallel branches via RunnableParallel:
    #   "context" branch: query → retriever → format_docs → string of chunks
    #   "question" branch: query passes through unchanged (RunnablePassthrough)
    # Both feed into the prompt template, then the LLM, then string extraction.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | hr_n_it_prompt | llm | StrOutputParser()
    )

    # ─── Step 5a: Query Rewriter ───
    # Resolves references like "that ticket" using prior state (last_ticket_id) and history.
    def rewrite_query(state):
        original = state["query"]
        last_id = (state.get("last_ticket_id") or "").strip()
        effective = original
        if last_id:
            lo = original.lower()
            refers = ("that ticket" in lo) or ("that" in lo and ("ticket" in lo or "priority" in lo or "status" in lo or "description" in lo)) or ("it" in lo and ("ticket" in lo or "priority" in lo or "status" in lo or "description" in lo))
            if refers and ("tick-" not in lo):
                # Lightweight rewrite: append explicit reference
                effective = f"{original} (refers to ticket {last_id})"
        return {"effective_query": effective}

    # ─── Step 5: Define the Supervisor (intent classifier) ───
    # The supervisor is the entry-point node after rewrite. It makes a single LLM call
    # to classify into one of three intents.
    classification_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intent classifier. Read the customer's query and return exactly one of: \n"
         "- \"kb_query\" — general questions about HR/Onboarding/Payroll or general IT information; no request to create a ticket and not asking for ticket status.\n"
         "- \"tkct_creation\" — IT support issues or problem reports requesting help, a fix, or to create/open/raise a ticket (e.g., app crashing, cannot login, errors).\n"
         "- \"tkct_status\" — requests about the status/updates of an existing ticket (e.g., includes an ID like TICK-002 or asks about \"ticket status\").\n"
         "Rules:\n"
         "1) If the user asks for ticket status, return tkct_status.\n"
         "2) Else if the user reports an IT support issue and wants help or a ticket created, return tkct_creation.\n"
         "3) Otherwise return kb_query.\n"
         "Return only the label: kb_query, tkct_creation, or tkct_status."),
        ("human", "{query}"),
    ])

    def classify_intent(state):
        """Supervisor node: classifies intent and writes it to state.

        Rule-first, LLM-second: if a ticket ID like TICK-002 is present, route
        deterministically to tkct_status. Otherwise, fall back to LLM.
        """
        raw_query = state.get("effective_query") or state["query"]
        # Normalize Unicode and dash variants to improve robustness
        norm_q = unicodedata.normalize("NFKC", raw_query)
        norm_q = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", norm_q)
        if re.search(r"\bTICK-\s*\d+\b", norm_q, re.IGNORECASE):
            intent = "tkct_status"
        else:
            chain = classification_prompt | llm | StrOutputParser()
            intent = chain.invoke({"query": raw_query}).strip().lower()
            if intent not in ("kb_query", "tkct_creation", "tkct_status"):
                intent = "kb_query"
        print("-"*50)
        print(f"    intent --> {intent}")
        print("-"*50)
        return {"intent": intent}

    # ─── Step 6: Define the RAG Agent ───
    # Handles "kb_query" intent.  Retrieves relevant chunks from the vector store,
    # feeds them as context to the LLM, and returns a grounded answer.
    def kb_query_agent(state):
        """RAG Agent node: answers HR & IT questions using RAG.

        When `enable_reranking` is True , it over-fetches
        documents and re-sorts by relevance score to improve retrieval quality
        (measured via MRR — Mean Reciprocal Rank).
        """
        question = state.get("effective_query") or state["query"]
        if enable_reranking:
            # Over-fetch candidate docs, then use the LLM to re-score each
            # one against the query and keep only the top_k most relevant
            fetch_k = rerank_fetch_k or top_k * 2
            candidate_docs = vectorstore.similarity_search(question, k=fetch_k)

            # LLM-as-reranker: ask the model to score each doc's relevance 0-10
            rerank_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a relevance scoring system. Given a query and a document, "
                 "rate how relevant the document is to answering the query on a scale "
                 "of 0 to 10. Respond with ONLY a single integer."),
                ("human",
                 "Query: {query}\n\nDocument:\n{document}\n\nRelevance score (0-10):"),
            ])
            rerank_chain = rerank_prompt | llm | StrOutputParser()

            scored = []
            for doc in candidate_docs:
                score_str = rerank_chain.invoke({
                    "query": question,
                    "document": doc.page_content,
                })
                try:
                    score = int(score_str.strip())
                except ValueError:
                    score = 0
                scored.append((doc, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            retrieved_docs = [doc for doc, _ in scored[:top_k]]
        else:
            # Standard retrieval — returns top_k chunks by vector similarity
            retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)
        # Track which source files were used (for observability / evaluation)
        sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
        # Note: we call the prompt→LLM chain directly (not rag_chain) because
        # we already fetched the docs above and need the sources list.
        chain = hr_n_it_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {
            "response": answer,
            "context": context,
            "retrieved_sources": sources,
        }

    # ─── Step 7: Define the Ticket Status Agent ───
    # Handles "tkct_status" intent.  Looks up mock ticket data by ID and
    # uses the LLM to formulate a natural-language summary.
    ticket_status_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support agent helping with ticket inquiries .\n"
         "Answer based ONLY on the ticket data provided. Be friendly and concise.\n"
         "IMPORTANT: Never reveal the customer's SSN (even last 4 digits) in your response."),
        ("human",
         "Ticket data:\n{ticket_data}\n\nCustomer question: {question}"),
    ])

    def ticket_status_agent(state):
        """Ticket Agent node: looks up ticket details from the mock database.
        Flow:
          1. Extract ticket ID (TICK-XXXXX) from the query using regex
          2. Look up the ticket in MOCK_TICKETS
          3. Ask the LLM to compose a friendly summary of the account data
        """
        raw_query = state.get("effective_query") or state["query"]
        # Normalize Unicode (e.g., fancy dashes) and search robustly for ticket IDs
        query = unicodedata.normalize("NFKC", raw_query)
        # Replace common dash variants with ASCII hyphen
        query = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", query)
        match = re.search(r"\bTICK-\s*(\d+)\b", query, re.IGNORECASE)
        if not match:
            # No ticket number found — ask the user to provide one
            return {
                "response": (
                    "I'd be happy to help with your ticket! Could you please "
                    "provide your ticket number? It starts with 'TICK-' followed "
                    "by digits (e.g., TICK-005)."
                ),
                "context": "",
                "retrieved_sources": [],
            }
        # Normalize to canonical form like 'TICK-002'
        digits = match.group(1)
        ticket_id = f"TICK-{digits}".upper()
        # Resolve tickets dataset path alongside this module
        tickets_dir = Path(__file__).parent / "tickets"
        ticket_file = tickets_dir / "synthetic_tickets.json"

        # Load our support ticket dataset (list of ticket objects). Fallback to the
        # in-module MOCK_TICKETS dict if the JSON file isn't available.
        tickets_by_id = {}
        try:
            with open(ticket_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                # Build a fast lookup by full ticket_id like "TICK-002"
                tickets_by_id = {
                    str(item.get("ticket_id", "")).upper(): item for item in data if isinstance(item, dict)
                }
            elif isinstance(data, dict):
                # If someone changes the JSON to a dict later, accept it too
                tickets_by_id = {k.upper(): v for k, v in data.items()}
        except Exception:
            # Silent fallback to bundled mock tickets if file isn't readable
            tickets_by_id = {k.upper(): v for k, v in MOCK_TICKETS.items()}

        ticket = tickets_by_id.get(ticket_id)
        if not ticket:
            # Ticket ID parsed but doesn't exist in our mock data
            return {
                "response": (
                    f"I couldn't find ticket {ticket_id} in our system. "
                    "Please double-check or contact support@securebank.com."
                ),
                "context": "",
                "retrieved_sources": [],
            }

        context = json.dumps(ticket, indent=2)
        chain = ticket_status_prompt | llm | StrOutputParser()
        response = chain.invoke({"ticket_data": context, "question": query})
        return {"response": response, "context": context, "retrieved_sources": []}

    # ─── Step 8: Define the Escalation Agent ───
    # Handles "tkct_creation" intent.  Unlike the other agents, this one does NOT
    # perform any retrieval — it simply generates directs the customer to human support channels
    # who after getting confirmation , creates ticket .
    ticket_creation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior customer support agent at SecureBank handling an escalation.\n"
         "1. Summarize their issue briefly\n"
         "2. Ask them to share Email , Issue description, summary, category and priority\n"
         "3. Let them know a senior specialist will follow up\n"
         "4. Provide: support@securebank.com or 1-800-555-0199\n\n"
         "Be warm, professional, and concise. "
         "Do NOT try to solve the problem yourself."),
        ("human", "{query}"),
    ])

    def ticket_creation_agent(state):
        """Ticket Creation Agent node: fetches Email , 
        Issue description, summary, category and priority.
        No retrieval or database lookup — the LLM generates a compassionate
        acknowledgment and routes the customer to live support channels.
        """
        chain = ticket_creation_prompt | llm | StrOutputParser()
        response = chain.invoke({"query": state["query"]})
        return {"response": response, "context": "", "retrieved_sources": []}

    # Build agent nodes from external modules
    kb_node_name, kb_node = make_kb_agent(
        llm=llm,
        retriever=retriever,
        vectorstore=vectorstore,
        hr_n_it_prompt=hr_n_it_prompt,
        format_docs=format_docs,
        top_k=top_k,
        enable_reranking=enable_reranking,
        rerank_fetch_k=rerank_fetch_k,
    )
    ts_node_name, ts_node = make_ticket_status_agent(
        llm=llm,
        tickets_dir=Path(__file__).parent / "tickets",
        fallback_tickets=MOCK_TICKETS,
    )
    tc_node_name, tc_node = make_ticket_creation_agent(llm=llm)

    # ─── Step 9: Define routing logic ───
    # Maps the intent string (set by the supervisor) to the corresponding agent
    # node name.  Falls back to "kb_query_agent" for any unknown intent.
    def route_by_intent(state) -> Literal[
        "kb_query_agent", "ticket_creation_agent", "ticket_status_agent"
    ]:
        return {
            "kb_query": kb_node_name,
            "tkct_status": ts_node_name,
            "tkct_creation": tc_node_name,
        }.get(state["intent"], kb_node_name)

    # ─── Step 10: Assemble the LangGraph ───
    # Graph structure:
    #
    #   START → classify_intent ─┬─ "knowlegeBase"          → kb_query     → END
    #                             ├─ "reading"  → tkct_status    → END
    #                             └─ "escalation"      → tkct_creation → END
    #
    graph = StateGraph(SupportState)
    graph.add_node("rewrite_query", rewrite_query)        # Resolve anaphora / references
    graph.add_node("classify_intent", classify_intent)    # Supervisor
    graph.add_node(kb_node_name, kb_node)                  # RAG over hr&it docs
    graph.add_node(ts_node_name, ts_node)                  # Mock DB lookup
    graph.add_node(tc_node_name, tc_node)                  # Empathetic handoff

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "classify_intent")
    # Conditional edges: the supervisor's output determines which agent runs next
    graph.add_conditional_edges("classify_intent", route_by_intent)
    # Each agent terminates the graph after producing a response
    graph.add_edge(kb_node_name, END)
    graph.add_edge(ts_node_name, END)
    graph.add_edge(tc_node_name, END)

    try:
        # ─── Enable per-thread memory via LangGraph checkpointer ───
        # Option A: in-memory (process-lifetime)
        from langgraph.checkpoint.memory import MemorySaver
        saver = MemorySaver()

        # Option B: durable SQLite (recommended)
        # from langgraph.checkpoint.sqlite import SqliteSaver
        # db_path = Path(__file__).parent / "data" / "conversations.db"
        # db_path.parent.mkdir(parents=True, exist_ok=True)
        # saver = SqliteSaver(str(db_path))

        app = graph.compile(checkpointer=saver)

        # Return all key components so modules can access them individually.
        # For example, Module B uses `retriever` and `vectorstore` for MRR evaluation,
        # and Module D uses `llm` for token counting.
        return {
            "app": app,              # The compiled LangGraph — pass to ask()
            "retriever": retriever,  # Chroma retriever for direct retrieval tests
            "format_docs": format_docs,  # Doc formatter (used in evaluation)
            "llm": llm,              # Local TinyLlama instance (shared across agents)
            "rag_chain": rag_chain,  # Standalone RAG chain (retriever → LLM)
            "vectorstore": vectorstore,  # Raw Chroma store (for similarity_search_with_relevance_scores)
        }
    except Exception:
        raise


def ask(app, query: str, thread_id: str | None = None) -> dict:
    """Invoke the multi-agent system with optional per-thread memory.

    If thread_id is provided, the LangGraph checkpointer resumes prior state
    for that thread. IMPORTANT: Do not send default values for other fields,
    or they will overwrite persisted state. Only send the new query.
    """
    payload = {"query": query}
    cfg = {"configurable": {"thread_id": thread_id}} if thread_id else None
    return app.invoke(payload, config=cfg) if cfg else app.invoke(payload)
