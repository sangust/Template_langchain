import json
import logging
import asyncio
from functools import partial
 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
 
from app.src.config.settings import settings, ollama_system_prompt
from app.src.providers.llm_provider import get_llm
from app.src.providers.redis_provider import get_redis_client
 
logger = logging.getLogger(__name__)

def _build_messages(
    history: list[dict],
    system_prompt: str,
    user_message: str,
) -> list[BaseMessage]:
    """
    Converte o histórico (lista de dicts {role, content}) em objetos LangChain
    e monta a lista final de mensagens para o LLM.
 
    Estrutura esperada pelo Ollama/LangChain:
        SystemMessage  ← aparece uma única vez, no início
        HumanMessage   ← turno do usuário
        AIMessage      ← turno do assistente
        ...
        HumanMessage   ← mensagem atual
    """
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
 
    for entry in history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        # roles desconhecidos são ignorados silenciosamente
 
    messages.append(HumanMessage(content=user_message))
    return messages



async def chat(
    message: str,
    history: list | None = None
) -> dict:
    """
    Envia uma mensagem ao LLM e retorna a resposta.

    Args:
        message: Mensagem do usuário
        history: Histórico anterior como lista de dicts {role, content}.

    Returns:
        dict com 'answer' (str) e 'model_used' (str).
    """
    system_prompt = ollama_system_prompt
    model_name = settings.ollama_default_model
    
    loop = asyncio.get_event_loop()
    messages = _build_messages(history or [], system_prompt, message)
    llm = get_llm(model=model_name)
    response = await loop.run_in_executor(None, partial(llm.invoke, messages))
    return {"answer": response.content, "model_used": model_name}


def add_message(session_id, role, content):
    try:
        client = get_redis_client()
        key = f"{settings.redis_namespace}:{session_id}"
        with client.pipeline() as pipe:
            pipe.rpush(key, json.dumps({"role": role, "content": content}))
            pipe.ltrim(key, -20, -1)
            pipe.expire(key, 1800)
            pipe.execute()
    except Exception as exc:
        logger.warning("Falha ao salvar no Redis: %s", exc)
 
 
def get_history(session_id: str) -> list[dict]:
    try:
        client = get_redis_client()
        key = f"{settings.redis_namespace}:{session_id}"
        raw = client.lrange(key, 0, -1)

        history = []
        for entry in raw:
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8")
            history.append(json.loads(entry))

        return history

    except Exception as exc:
        logger.warning("Falha ao ler histórico do Redis: %s", exc)
        return []


