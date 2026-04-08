from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.src.schemas.chat_schema import ChatRequest, ChatResponse
from app.src.services.chat_service import chat, add_message, get_history
from app.src.config.logging import get_logger
from app.src.services.session_service import get_session_id
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

home = APIRouter(tags=["Home"])
templates = Jinja2Templates(directory="app/src/templates")


@home.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Retorna a página inicial da UI."""
    logger.debug("home_page_requested", client=request.client.host if request.client else "unknown")
    return templates.TemplateResponse(request, "home.html", {"request": request})


@home.post("/", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest, response: Response) -> ChatResponse:
    """
    Envia uma mensagem ao LLM e retorna a resposta.

    Você pode customizar:
    - **message**: conteúdo da mensagem
    - **use_rag**: ativar RAG (opcional)
    """
    session_id = None
    
    try:
        session_id = get_session_id(request, response)
        client_ip = request.client.host if request.client else "unknown"
        
        logger.info(
            "chat_endpoint_request",
            session_id=session_id,
            client_ip=client_ip,
            message_length=len(body.message)
        )
        
        # Recuperar histórico
        history = get_history(session_id)
        
        # Chamar serviço de chat
        result = await chat(
            message=body.message,
            history=history,
            session_id=session_id
        )
        
        # Salvar mensagens no histórico
        add_message(session_id, "user", body.message)
        add_message(session_id, "assistant", result["answer"])
        
        logger.info(
            "chat_endpoint_success",
            session_id=session_id,
            model=result["model_used"],
            response_length=len(result["answer"])
        )
        
        return ChatResponse(**result)
        
    except ValueError as e:
        logger.warning(
            "chat_endpoint_validation_error",
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
        
    except ConnectionError as e:
        logger.error(
            "chat_endpoint_connection_error",
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=503,
            detail="LLM service temporarily unavailable"
        )
        
    except TimeoutError as e:
        logger.error(
            "chat_endpoint_timeout",
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=504,
            detail="LLM response timeout"
        )
        
    except Exception as e:
        logger.error(
            "chat_endpoint_unexpected_error",
            session_id=session_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )