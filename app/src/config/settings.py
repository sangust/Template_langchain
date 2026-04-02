from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configurações gerais da aplicação.
    Todas as variáveis podem ser sobrescritas via arquivo .env
    """

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "qwen3:4b-instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = "Voce e um assistente."
    ollama_num_ctx: int = 2048

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # --- RAG (opcional) ---
    rag_enabled: bool = False
    rag_docs_path: str = "docs"          # pasta com os PDFs/TXTs para indexar
    rag_collection_name: str = "default" # nome da coleção no ChromaDB
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 4                   # quantos chunks recuperar por consulta


    #Pegar as configurações do arquivo .env automaticamente
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instância global — inicializada uma única vez na subida da aplicação
settings = Settings()
