from pathlib import Path
import json
import re
import unicodedata
from typing import Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Factory for the Ticket Status agent node
# Returns (node_name, node_fn)

def make_ticket_status_agent(
    llm,
    tickets_dir: Path | None = None,
    fallback_tickets: dict | None = None,
) -> tuple[str, Callable[[dict], dict]]:
    """Create the ticket status agent node.

    tickets_dir points to the folder containing synthetic_tickets.json.
    fallback_tickets is a dict used if the JSON file cannot be read.
    """

    ticket_status_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support agent helping with ticket inquiries .\n"
         "Answer based ONLY on the ticket data provided. Be friendly and concise.\n"
         "IMPORTANT: Never reveal the customer's SSN (even last 4 digits) in your response."),
        ("human",
         "Ticket data:\n{ticket_data}\n\nCustomer question: {question}"),
    ])

    def ticket_status_agent(state: dict) -> dict:
        raw_query = state.get("effective_query") or state["query"]
        base_history = state.get("history", [])
        entry_user = {"role": "user", "content": raw_query}
        # Normalize Unicode (e.g., fancy dashes) and search robustly for ticket IDs and emails
        query = unicodedata.normalize("NFKC", raw_query)
        query = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", query)
        id_match = re.search(r"\bTICK-\s*(\d+)\b", query, re.IGNORECASE)
        email_match = re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", query, re.IGNORECASE)

        # Resolve dataset path
        tdir = tickets_dir or Path(__file__).resolve().parent
        ticket_file = tdir / "synthetic_tickets.json"

        # Load dataset into both dict-by-id and list for filtering by email
        tickets_by_id: dict[str, dict] = {}
        tickets_list: list[dict] = []
        try:
            with open(ticket_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                tickets_list = [it for it in data if isinstance(it, dict)]
                tickets_by_id = {str(it.get("ticket_id", "")).upper(): it for it in tickets_list}
            elif isinstance(data, dict):
                tickets_by_id = {k.upper(): v for k, v in data.items()}
                tickets_list = list(tickets_by_id.values())
        except Exception:
            if fallback_tickets:
                tickets_by_id = {k.upper(): v for k, v in fallback_tickets.items()}
                tickets_list = list(tickets_by_id.values())

        # 1) Prefer explicit ticket ID when present
        if id_match:
            digits = id_match.group(1)
            ticket_id = f"TICK-{digits}".upper()
            ticket = tickets_by_id.get(ticket_id)
            if not ticket:
                resp = (
                    f"I couldn't find ticket {ticket_id} in our system. "
                    "Please double-check or contact support@securebank.com."
                )
                return {
                    "response": resp,
                    "context": "",
                    "retrieved_sources": [],
                    "history": base_history + [entry_user, {"role": "assistant", "content": resp}],
                }
            context = json.dumps(ticket, indent=2)
            chain = ticket_status_prompt | llm | StrOutputParser()
            response = chain.invoke({"ticket_data": context, "question": raw_query})
            return {
                "response": response,
                "context": context,
                "retrieved_sources": ["tickets.db"],
                "history": base_history + [entry_user, {"role": "assistant", "content": response}],
                "last_ticket_id": ticket_id,
            }

        # 2) Otherwise, try lookup by email address (created_for)
        if email_match:
            email = email_match.group(0).lower()
            matches = [t for t in tickets_list if str(t.get("created_for", "")).lower() == email]
            if not matches:
                resp = (
                    f"I couldn't find any tickets for {email}. "
                    "Please verify the email or provide your ticket ID (e.g., TICK-005)."
                )
                return {
                    "response": resp,
                    "context": "",
                    "retrieved_sources": [],
                    "history": base_history + [entry_user, {"role": "assistant", "content": resp}],
                }
            # Sort newest first by created_date (YYYY-MM-DD lexicographically sorts correctly)
            try:
                matches.sort(key=lambda t: str(t.get("created_date", "")), reverse=True)
            except Exception:
                pass
            primary = matches[0]
            context = json.dumps(primary, indent=2)
            chain = ticket_status_prompt | llm | StrOutputParser()
            base = chain.invoke({"ticket_data": context, "question": raw_query})
            if len(matches) == 1:
                response = base
            else:
                others = ", ".join([str(t.get("ticket_id", "")) for t in matches[1:5]])
                note = f"\n\nNote: Found {len(matches)} tickets for {email}. Showing the most recent ({primary.get('ticket_id','')}). Other tickets: {others}."
                response = base + note
            return {
                "response": response,
                "context": context,
                "retrieved_sources": ["tickets.db"],
                "history": base_history + [entry_user, {"role": "assistant", "content": response}],
                "last_ticket_id": str(primary.get("ticket_id", "")).upper(),
            }

        # 3) Neither ticket ID nor email provided — ask for one
        resp = (
            "I'd be happy to help with your ticket! Please provide your ticket number "
            "(e.g., TICK-005) or the email address used to create the ticket."
        )
        return {
            "response": resp,
            "context": "",
            "retrieved_sources": [],
            "history": base_history + [entry_user, {"role": "assistant", "content": resp}],
        }

    return "ticket_status_agent", ticket_status_agent
