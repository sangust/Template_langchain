from redis import ConnectionPool, Redis
from app.src.config.settings import settings

# Pool criado uma única vez ao importar o módulo.
# Todas as chamadas a get_redis_client() reusam conexões do mesmo pool,
# sem abrir/fechar sockets a cada requisição.
_pool = ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    decode_responses=True,
    max_connections=10
)


def get_redis_client() -> Redis:
    """Retorna um cliente Redis que usa o pool de conexões compartilhado."""
    return Redis(connection_pool=_pool)