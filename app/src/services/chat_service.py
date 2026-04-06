from langchain_core.messages import HumanMessage, SystemMessage
from app.src.providers.llm_provider import get_llm
from app.src.config.settings import settings, ollama_system_prompt
from langchain_ollama import ChatOllama
from app.src.providers.redis_provider import get_redis_client
import json



def chat(
    message: str,
    model: str | None = None,
    system_prompt: str = ollama_system_prompt,
    history: list | None = None,
    temperature: float | None = settings.ollama_temperature,
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
    
    model_name = model or settings.ollama_default_model
    try:
        llm: ChatOllama = get_llm(model=model_name)
    except Exception as e:
        print("Error loading LLM:", e)
        raise e
    
    # Monta as mensagens — LangChain usa essa estrutura para todos os LLMs
    messages = []
    if history:
        messages.extend(history)

    messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=message))
    
    response = llm.invoke(messages, config={"configurable": {"temperature": temperature}})

    return {
        "answer": response.content,
        "model_used": model_name,
    }


def add_message(session_id, role, content):
    redis_client = get_redis_client()
    key = f"chat:{session_id}"
    redis_client.rpush(key, json.dumps({"role": role, "content": content}))
    redis_client.ltrim(key, -20, -1)   # mantém últimas 20 mensagens
    redis_client.expire(key, 1800)     # expira em 30 minutos


def get_history(session_id):
    redis_client = get_redis_client()
    key = f"chat:{session_id}"
    data = redis_client.lrange(key, 0, -1)
    return [json.loads(x) for x in data]