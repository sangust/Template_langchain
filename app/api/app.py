from fastapi import FastAPI
from prometheus_client import make_asgi_app
from app.api.routes.home import home
from app.src.config.settings import settings
from app.src.config.logging import configure_logging, get_logger
from app.api.middleware import http_logging_middleware as logging_middleware

# Inicializar logging
env = settings.environment if hasattr(settings, 'environment') else 'development'
configure_logging(log_level="INFO", env=env)
logger = get_logger(__name__)

# Criar aplicação FastAPI
app = FastAPI(
    title="Template LangChain",
    description="Template reutilizável para projetos com LangChain e modelos locais via Ollama",
    version="1.0.0",
)

# Adicionar middleware de logging
@app.middleware("http")
async def http_logging_middleware(request, call_next):
    return await logging_middleware(request, call_next)

# Montar Prometheus metrics em /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Rotas
app.include_router(home)

@app.on_event("startup")
async def startup_event():
    logger.info("app_startup", version="1.0.0", environment=env)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("app_shutdown")