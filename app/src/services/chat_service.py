from langchain_core.messages import HumanMessage, SystemMessage
from app.src.providers.llm_provider import get_llm
from app.src.config.settings import settings


def chat(
    message: str,
    model: str | None = None,
    system_prompt: str = settings.system_prompt,
    temperature: float | None = None,
) -> dict:
    """
    Envia uma mensagem ao LLM e retorna a resposta.

    Args:
        message: Mensagem do usuário
        model: Modelo a usar (opcional)
        system_prompt: System prompt customizado (opcional)
        temperature: Temperatura (opcional)

    Returns:
        dict com 'answer' e 'model_used'
    """
    
    model_name = model or settings.default_model
    llm = get_llm(model=model_name, temperature=temperature)
    
    # Monta as mensagens — LangChain usa essa estrutura para todos os LLMs
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message),
    ]
    
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "model_used": model_name,
    }