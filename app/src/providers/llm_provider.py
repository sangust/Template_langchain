from langchain_ollama import ChatOllama
from app.src.config.settings import settings, ollama_system_prompt


def get_llm(
    model: str | None = None
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
    llm_model = model if model is not None else settings.ollama_default_model

    return ChatOllama(
        model=llm_model,
        temperature=settings.ollama_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        system=ollama_system_prompt,
        flash_attention=settings.ollama_flash_attention,
    )
