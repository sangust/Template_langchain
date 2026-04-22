from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.src.schemas.chat_schema import ChatRequest, ChatResponse
from app.src.services.chat_service import chat, add_message, get_history
from app.src.config.logging import get_logger
from app.src.services.session_service import get_session_id
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi.responses import StreamingResponse

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

home = APIRouter(tags=["Home"])
templates = Jinja2Templates(directory="app/src/templates")


@home.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Retorna a página inicial da UI."""
    logger.debug("home_page_requested", client=request.client.host if request.client else "unknown")
    return templates.TemplateResponse(request, "home.html", {"request": request})


@home.post("/")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest, response: Response):

    session_id = None

    try:
        session_id = get_session_id(request, response)
        history = get_history(session_id)

        add_message(session_id, "user", body.message)

        async def generate():
            parts = []

            try:
                async for chunk in chat(
                    message=body.message,
                    history=history,
                    session_id=session_id,
                    use_rag=body.use_rag
                ):
                    parts.append(chunk)

                    # SSE correto
                    yield f"data: {chunk}\n\n"

                # sinal opcional de fim
                yield "event: end\ndata: [DONE]\n\n"

            except asyncio.CancelledError:
                logger.warning(
                    "client_disconnected",
                    session_id=session_id
                )
                raise

            finally:
                if parts:
                    full_message = "".join(parts)
                    add_message(session_id, "assistant", full_message)

                    logger.info(
                        "chat_persisted",
                        session_id=session_id,
                        response_length=len(full_message)
                    )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except ConnectionError:
        raise HTTPException(status_code=503, detail="LLM unavailable")

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout")

    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")