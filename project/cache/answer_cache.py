from threading import RLock
from typing import TypedDict


class CachedAnswer(TypedDict):
    response: str
    intent: str
    retrieved_sources: list[str]


cache: dict[tuple[str, str], CachedAnswer] = {}
cache_lock = RLock()


def _normalize_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def get_cached_answer(
    conversation_id: str, question: str
) -> CachedAnswer | None:
    key = (conversation_id, _normalize_question(question))
    with cache_lock:
        cached = cache.get(key)
        if cached is None:
            return None
        return {
            **cached,
            "retrieved_sources": list(cached["retrieved_sources"]),
        }


def cache_answer(
    conversation_id: str,
    question: str,
    response: str,
    intent: str,
    retrieved_sources: list[str],
) -> None:
    if not response:
        return

    key = (conversation_id, _normalize_question(question))
    entry: CachedAnswer = {
        "response": response,
        "intent": intent,
        "retrieved_sources": list(retrieved_sources),
    }
    with cache_lock:
        cache[key] = entry
