from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.src.schemas.chat_schema import ChatRequest, ChatResponse
from app.src.services.chat_service import chat
from app.src.config.settings import settings
import traceback

home = APIRouter(tags=["Home"])
templates = Jinja2Templates(directory="app/src/templates")


@home.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse(request, "home.html", {"request": request})

@home.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Envia uma mensagem ao LLM e retorna a resposta.
 
    Você pode customizar:
    - **model**: qual modelo usar
    """
    try:
        result = chat(
            message=request.message,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
        )
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
        
