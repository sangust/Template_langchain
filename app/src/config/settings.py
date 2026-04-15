from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Configurações gerais da aplicação.
    Todas as variáveis podem ser sobrescritas via arquivo .env
    """
    # --- Environment ---
    environment: str = "development"
    
    # --- Ollama ---
    ollama_default_model: str = "qwen3:4b-instruct"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://ollama:11434"

    ollama_system_prompt_path: str = "app/src/prompts/system.md"
    

    ollama_temperature: float = 0.7
    ollama_max_tokens: int = 4096
    ollama_num_ctx: int = 4096
    ollama_keep_alive: str = "5m"
    ollama_flash_attention: bool = True

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_namespace: str = "historyChat"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # --- RAG (opcional) ---
    rag_enabled: bool = False
    rag_docs_path: str = "app/src/docs"          # pasta com os PDFs/TXTs para indexar
    rag_collection_name: str = "default" # nome da coleção no ChromaDB
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 4                   # quantos chunks recuperar por consulta
    
    # ChromaDB
    chroma_host: str = "chroma"
    chroma_port: int = 8000

    #Pegar as configurações do arquivo .env automaticamente
    model_config = SettingsConfigDict(env_file="ollama.env", env_file_encoding="utf-8")


# Instância global — inicializada uma única vez na subida da aplicação
settings = Settings()

try:
    ollama_system_prompt = Path(settings.ollama_system_prompt_path).read_text(encoding="utf-8")
except FileNotFoundError:
    ollama_system_prompt = "Você é um assistente útil."
