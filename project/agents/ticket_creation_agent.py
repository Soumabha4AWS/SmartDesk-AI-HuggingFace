from typing import Callable, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Factory for the Ticket Creation agent node
# Returns (node_name, node_fn)

def make_ticket_creation_agent(llm) -> tuple[str, Callable[[dict], dict]]:
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

    def ticket_creation_agent(state: dict) -> dict:
        chain = ticket_creation_prompt | llm | StrOutputParser()
        response = chain.invoke({"query": state["query"]})
        base_history = state.get("history", [])
        entry_user = {"role": "user", "content": state.get("query", "")}
        entry_assistant = {"role": "assistant", "content": response}
        return {
            "response": response,
            "context": "",
            "retrieved_sources": [],
            "history": base_history + [entry_user, entry_assistant],
        }

    return "ticket_creation_agent", ticket_creation_agent
