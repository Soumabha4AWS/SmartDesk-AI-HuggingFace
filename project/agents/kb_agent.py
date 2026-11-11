from typing import Callable, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Factory for the RAG/KB agent node
# Returns (node_name, node_fn)

def make_kb_agent(
    llm,
    retriever,
    vectorstore,
    hr_n_it_prompt: ChatPromptTemplate,
    format_docs: Callable[[list], str],
    top_k: int = 3,
    enable_reranking: bool = False,
    rerank_fetch_k: int | None = None,
) -> tuple[str, Callable[[dict], dict]]:
    """Create the KB/RAG agent node using the provided dependencies."""

    def kb_query_agent(state: dict) -> dict:
        question = state.get("effective_query") or state["query"]
        if enable_reranking:
            # Over-fetch candidate docs, then use the LLM to re-score each
            fetch_k = rerank_fetch_k or top_k * 2
            candidate_docs = vectorstore.similarity_search(question, k=fetch_k)

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
            retrieved_docs = retriever.invoke(question)

        context = format_docs(retrieved_docs)
        sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
        chain = hr_n_it_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        base_history = state.get("history", [])
        entry_user = {"role": "user", "content": question}
        entry_assistant = {"role": "assistant", "content": answer}
        return {
            "response": answer,
            "context": context,
            "retrieved_sources": sources,
            "history": base_history + [entry_user, entry_assistant],
        }

    return "kb_query_agent", kb_query_agent
