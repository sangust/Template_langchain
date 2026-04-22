import asyncio
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from app.src.services.rag_service import retrieve_context
from app.src.config.settings import settings, ollama_system_prompt
from app.src.config.logging import get_logger
from app.src.config.metrics import (
    chat_response_time_seconds,
    chat_errors_total,
    ollama_requests_total,
)
from app.src.providers.llm_provider import get_llm

logger = get_logger(__name__)


def _build_messages(
    history: list[dict],
    system_prompt: str,
    user_message: str,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    for entry in history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))
    return messages


async def chat(
    message: str,
    history: list | None = None,
    session_id: str = "unknown",
    use_rag: bool = False
) -> AsyncGenerator[str, None]:

    system_prompt = ollama_system_prompt
    model_name = settings.ollama_default_model

    try:
        if use_rag:
            history = (history or [])[-2:]
            context_text = await retrieve_context(message)
            if context_text:
                message = f"Contexto:\n{context_text}\n\nPergunta:\n{message}"

        messages = _build_messages(history or [], system_prompt, message)
        llm = get_llm(model=model_name)

        loop = asyncio.get_running_loop()
        start_time = loop.time()

        parts = []

        async for chunk in llm.astream(messages):
            if chunk.content:
                parts.append(chunk.content)
                yield chunk.content

        duration = loop.time() - start_time

        chat_response_time_seconds.labels(model=model_name).observe(duration)
        ollama_requests_total.labels(model=model_name, status="success").inc()

        logger.info(
            "chat_response_completed",
            session_id=session_id,
            model=model_name,
            response_length=len("".join(parts)),
            duration_seconds=round(duration, 2)
        )

    except asyncio.CancelledError:
        logger.warning(
            "chat_stream_cancelled",
            session_id=session_id,
            model=model_name
        )
        ollama_requests_total.labels(model=model_name, status="cancelled").inc()
        raise

    except ConnectionError as e:
        ollama_requests_total.labels(model=model_name, status="error").inc()
        chat_errors_total.labels(error_type="connection").inc()
        raise

    except TimeoutError as e:
        ollama_requests_total.labels(model=model_name, status="timeout").inc()
        chat_errors_total.labels(error_type="timeout").inc()
        raise

    except Exception:
        ollama_requests_total.labels(model=model_name, status="error").inc()
        chat_errors_total.labels(error_type="unknown").inc()
        raise


def add_message(session_id: str, role: str, content: str) -> None:
    """
    Adiciona mensagem ao histórico Redis.
    
    Args:
        session_id: ID da sessão
        role: 'user' ou 'assistant'
        content: Conteúdo da mensagem
    """
    try:
        client = get_redis_client()
        key = f"{settings.redis_namespace}:{session_id}"
        
        start_time = asyncio.get_event_loop().time() if asyncio._get_running_loop() else 0
        
        with client.pipeline() as pipe:
            pipe.rpush(key, json.dumps({"role": role, "content": content}))
            pipe.ltrim(key, -20, -1)
            pipe.expire(key, 1800)
            pipe.execute()
        
        duration = asyncio.get_event_loop().time() - start_time if asyncio._get_running_loop() else 0
        
        logger.debug(
            "message_added_to_redis",
            session_id=session_id,
            role=role,
            duration_seconds=round(duration, 3)
        )
        
    except Exception as exc:
        logger.warning(
            "failed_to_save_message_redis",
            session_id=session_id,
            error=str(exc),
            exc_info=True
        )


def get_history(session_id: str) -> list[dict]:
    """
    Recupera histórico de mensagens do Redis.
    
    Args:
        session_id: ID da sessão
    
    Returns:
        Lista de mensagens: [{"role": "user", "content": "..."}, ...]
    """
    try:
        client = get_redis_client()
        key = f"{settings.redis_namespace}:{session_id}"
        raw = client.lrange(key, 0, -1)

        history = []
        for entry in raw:
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8")
            history.append(json.loads(entry))

        logger.debug(
            "history_retrieved",
            session_id=session_id,
            history_size=len(history)
        )
        
        return history

    except Exception as exc:
        logger.warning(
            "failed_to_retrieve_history",
            session_id=session_id,
            error=str(exc),
            exc_info=True
        )
        return []