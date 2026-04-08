import logging
import structlog
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def configure_logging(log_level: str = "INFO", env: str = "development"):
    """
    Configura logging estruturado com structlog + Python logging.
    
    Em desenvolvimento: logs em JSON no console
    Em produção: logs em arquivo com rotação + console
    """
    
    log_dir = Path("app/src/logs")

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Handler para arquivo em produção
    if env == "production":
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=10,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)
    
    
    structlog.configure(
        processors=[
            # Adiciona timestamp e level
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            
            # Adiciona informações do traceback se houver exceção
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            
            # Renderiza para JSON
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


def get_logger(name: str):
    """Obter logger estruturado para um módulo."""
    return structlog.get_logger(name)