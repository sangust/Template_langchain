from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.src.schemas.chat_schema import ChatRequest, ChatResponse
from app.src.services.chat_service import chat, add_message, get_history
from app.src.config.settings import settings, ollama_system_prompt
from app.src.services.session_service import get_session_id

home = APIRouter(tags=["Home"])
templates = Jinja2Templates(directory="app/src/templates")


@home.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request, "home.html", {"request": request})

@home.post("/", response_model=ChatResponse)
async def chat_endpoint(request: Request, body: ChatRequest, response: Response) -> ChatResponse:
    """
    Envia uma mensagem ao LLM e retorna a resposta.
 
    Você pode customizar:
    - **model**: qual modelo usar
    """
    try:
        session_id = get_session_id(request, response)
        history = get_history(session_id)
        history.append({"role": "user", "content": body.message})

        result = chat(
            message=body.message,
            system_prompt=ollama_system_prompt,
            history=history
        )
        add_message(session_id, "user", body.message)
        add_message(session_id, "assistant", result["answer"])
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
        
