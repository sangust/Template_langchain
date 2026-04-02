from langchain_openai import ChatOpenAI
from app.src.config.settings import settings


def get_llm(
    model: str | None = None,
    temperature: float | None = None
) -> ChatOpenAI:
    """
    Retorna uma instância do LLM conectada ao vLLM local.

    O vLLM expõe uma API compatível com OpenAI, então usamos
    ChatOpenAI apontando para o servidor local.

    Args:
        model: Nome do modelo. Se None, usa o DEFAULT_MODEL do .env
        temperature: Temperatura da geração. Se None, usa a do .env

    Returns:
        Instância configurada do ChatOpenAI
    """

    #pegando do env se não for passado nos parametros.
    llm_model = model if model is not None else settings.default_model
    llm_temperature = temperature if temperature is not None else settings.temperature
    
    return ChatOpenAI(
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=settings.max_tokens,
        base_url=settings.vllm_base_url,
        api_key="vllm-local",  # vLLM não precisa de key real, mas o campo é obrigatório
        # default_headers pode ser útil para autenticação futura
    )

     