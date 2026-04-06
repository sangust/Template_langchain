from fastapi import FastAPI
from app.src.api.routes.home import home
from app.src.config.settings import settings

app = FastAPI(
    title="Template LangChain com vLLM",
    description="Template reutilizável para projetos com LangChain e modelos locais via vLLM",
    version="1.0.0",
)

app.include_router(home)
