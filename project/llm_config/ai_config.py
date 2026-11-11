import os


def get_ai_router_client_config() -> tuple[str, str]:
    """Return the AI Router key and its OpenAI-compatible endpoint."""
    airouter_key = os.getenv("AIROUTER_API_KEY")
    if airouter_key:
        airouter_base_url = os.getenv("AIROUTER_BASE_URL")
        if not airouter_base_url:
            raise RuntimeError(
                "AIROUTER_API_KEY is set, but AIROUTER_BASE_URL is missing. "
                "Set it to the OpenAI-compatible API URL from your provider account."
            )
        return airouter_key, airouter_base_url

    raise RuntimeError(
        "AIROUTER_API_KEY is missing. Set AIROUTER_API_KEY and "
        "AIROUTER_BASE_URL in the repository-root .env file."
    )
