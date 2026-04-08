from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

http_requests_total = Counter(
    'http_requests_total',
    'Total de requisições HTTP',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'Duração das requisições HTTP em segundos',
    ['method', 'endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Requisições HTTP em progresso'
)


chat_messages_total = Counter(
    'chat_messages_total',
    'Total de mensagens processadas',
    ['session_id', 'role']  # role: user, assistant
)

chat_response_time_seconds = Histogram(
    'chat_response_time_seconds',
    'Tempo de resposta do LLM em segundos',
    ['model'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

chat_errors_total = Counter(
    'chat_errors_total',
    'Total de erros em chat',
    ['error_type']  # connection, timeout, etc
)


redis_operations_total = Counter(
    'redis_operations_total',
    'Total de operações Redis',
    ['operation', 'status']  # operation: get, set, etc | status: success, error
)

redis_operation_duration_seconds = Histogram(
    'redis_operation_duration_seconds',
    'Duração das operações Redis em segundos',
    ['operation'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5)
)

# MÉTRICAS DE LLM / OLLAMA
ollama_requests_total = Counter(
    'ollama_requests_total',
    'Total de requisições ao Ollama',
    ['model', 'status']  # status: success, timeout, error
)

ollama_response_time_seconds = Histogram(
    'ollama_response_time_seconds',
    'Tempo de resposta do Ollama em segundos',
    ['model'],
    buckets=(1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

ollama_tokens_generated = Counter(
    'ollama_tokens_generated',
    'Total de tokens gerados',
    ['model']
)


def track_function_time(metric_name: str, labels: dict = None):
    """
    Decorator para rastrear tempo de execução de função.
    
    Usage:
        @track_function_time('chat_response_time_seconds', {'model': 'qwen'})
        async def chat(message):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if labels:
                    metric_name.labels(**labels).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if labels:
                    metric_name.labels(**labels).observe(duration)
        
        return async_wrapper if hasattr(func, '__await__') else sync_wrapper
    
    return decorator