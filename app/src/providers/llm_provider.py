from langchain_ollama import ChatOllama
from app.src.config.settings import settings, ollama_system_prompt
from functools import lru_cache

@lru_cache(maxsize=8)
def get_llm(model: str | None = None) -> ChatOllama:
    """
    Retorna uma instância do LLM conectada ao Ollama local.
 
    A instância é cacheada por nome de modelo. A primeira chamada com um dado
    modelo cria o objeto, chamadas seguintes reutilizam a mesma instância sem
    re-inicializar o cliente HTTP.
 
    O system prompt NÃO é definido aqui: ele é adicionado como SystemMessage
    diretamente na lista de mensagens do chat_service, o que dá mais
    flexibilidade (um endpoint pode usar prompts diferentes sem precisar de
    novas instâncias do LLM).
 
    Args:
        model: Nome do modelo. Se None, usa OLLAMA_DEFAULT_MODEL do .env.
 
    Returns:
        Instância configurada do ChatOllama.
    """
    llm_model = model if model is not None else settings.ollama_default_model

    return ChatOllama(
        model=llm_model,
        temperature=settings.ollama_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        flash_attention=settings.ollama_flash_attention,
    )
