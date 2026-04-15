from functools import lru_cache
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.src.config.settings import settings
import chromadb

@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Inicializa o banco vetorial apenas quando chamado (Lazy Loading).
    Utiliza o padrão lru_cache para garantir que a coleção não seja recriada a cada requisição HTTP.
    """
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model, # Mantém coerência com o ecossistema local do template
        base_url=settings.ollama_base_url
    )
    chroma_client = chromadb.HttpClient(
        host=settings.chroma_host, # Ex: "chroma"
        port=settings.chroma_port  # Ex: 8000
    )

    return Chroma(
        client=chroma_client,
        collection_name=settings.rag_collection_name,
        embedding_function=embeddings
    )