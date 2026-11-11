import sys
from pathlib import Path
from dotenv import load_dotenv

# Load this repository's .env so the script works from any current directory.
load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[1] / ".env",
    override=True,
)
# Set up the system path to import project modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from cache.answer_cache import cache_answer, get_cached_answer
from smart_desk_ai_agent import build_support_agent, ask


print("Building Smart Desk AI agent...")
agent = build_support_agent(collection_name="fintech_demo")
app = agent["app"]
print("Pipeline ready.\n")

# For multi-conversation testing, allow each value to be either:
# - a plain string (single-shot, no memory), or
# - a (query, config_id) tuple where config_id is passed as thread_id.
TEST_QUERIES = {
    "CATEGORY 1 — TICKET Status (Thread A, step 1)": (
        "Give me the status for Ticket Id -- TICK-002", "cust-A"
    ),
    "CATEGORY 2 — TICKET Status (Thread A, step 2)": (
        "For whom was that ticket created", "cust-A"
    ),
    "CATEGORY 3 — TICKET Status (Thread B, step 1)": (
        "Is there any ticket avilable for Memory leak in background worker process", "cust-B"
    ),
    "CATEGORY 4 — Knowledge Base (Thread C, step 1)": (
        "How many leaves avilable in an year for my organization", "cust-C"
    ),
    "CATEGORY 5 — Knowledge Base (Thread D, step 1)": (
        "How do I reset my payroll password?", "cust-D"
    ),
    "CATEGORY 6 — TICKET Status (Thread E, step 1)": (
        "Give me the status for Ticket created for marry.smith@gmail.com", "cust-E"
    ),
    "CATEGORY 6 — TICKET Status (Thread E, step 2)": (
        "Give me the description and priority for that Ticket ", "cust-E"
    ),
    "CATEGORY 7 — KB Status (Thread F, step 1)": (
        "Please tell me Code of conduct in the organization ", "cust-F"
    )
}

print("=" * 60)



for category, item in TEST_QUERIES.items():
    # Support both str and (query, config_id) tuple
    if isinstance(item, tuple):
        query, config_id = item
    else:
        query, config_id = item, None
    print(f"  {category}")
    print(f"  Query: {query}")
    if config_id:
        print(f"  Thread: {config_id}")
    result = get_cached_answer(config_id, query) if config_id else None
    if result is None:
        result = ask(app, query, thread_id=config_id)
        if config_id:
            cache_answer(
                config_id,
                query,
                result.get("response", ""),
                result.get("intent", ""),
                result.get("retrieved_sources", []) or [],
            )
    response = result["response"]
    print(f"  Response: {response[:1000]}")
    print("*"*80)


print("""
Fintech Support Testing Ending .....
""")

print("=" * 60)
