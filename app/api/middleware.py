from fastapi import Request
import time
from app.src.config.logging import get_logger
from app.src.config.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_requests_in_progress
)

logger = get_logger(__name__)

async def http_logging_middleware(request: Request, call_next):
    """
    Middleware que registra todas as requisições HTTP com logs estruturados
    e coleta métricas de performance.
    """
    
    # Incrementar contador de requisições em progresso
    http_requests_in_progress.inc()
    
    # Capturar informações da requisição
    request_id = request.headers.get("X-Request-ID", "unknown")
    method = request.method
    path = request.url.path
    client_ip = request.client.host if request.client else "unknown"
    
    start_time = time.time()
    
    # Log de início
    logger.info(
        "http_request_started",
        request_id=request_id,
        method=method,
        path=path,
        client_ip=client_ip,
    )
    
    try:
        # Executar handler
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Log de sucesso
        logger.info(
            "http_request_completed",
            request_id=request_id,
            method=method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        
        # Métricas
        http_requests_total.labels(
            method=method,
            endpoint=path,
            status_code=response.status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=path
        ).observe(duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Log de erro
        logger.error(
            "http_request_failed",
            request_id=request_id,
            method=method,
            path=path,
            error=str(e),
            duration_ms=round(duration * 1000, 2),
            exc_info=True,
        )
        
        # Métricas
        http_requests_total.labels(
            method=method,
            endpoint=path,
            status_code=500
        ).inc()
        
        raise
        
    finally:
        # Decrementar contador de requisições em progresso
        http_requests_in_progress.dec()