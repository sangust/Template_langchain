from langchain_ollama import ChatOllama
from app.src.config.settings import settings


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOllama:
    """
    Retorna uma instância do LLM conectada ao Ollama local.

    O Ollama roda localmente e não precisa de API key.
    Por padrão usa o modelo definido em DEFAULT_MODEL no .env (qwen2.5:7b).

    Args:
        model: Nome do modelo. Se None, usa o DEFAULT_MODEL do .env
        temperature: Temperatura da geração. Se None, usa a do .env

    Returns:
        Instância configurada do ChatOllama
    """
    llm_model = model if model is not None else settings.default_model
    llm_temperature = temperature if temperature is not None else settings.temperature

    return ChatOllama(
        model=llm_model,
        temperature=llm_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
    )
