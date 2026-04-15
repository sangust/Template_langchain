import asyncio
from functools import partial
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from app.src.providers.vector_provider import get_vector_store
from app.src.config.settings import settings


async def retrieve_context(query: str) -> str:
    """
    Busca documentos semanticamente similares e retorna o texto concatenado.
    Executa em thread separada para não bloquear o event loop do FastAPI.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.rag_top_k})
    
    loop = asyncio.get_event_loop()
    # run_in_executor é mandatório aqui porque Chroma/LangChain usam I/O síncrono por baixo dos panos
    docs = await loop.run_in_executor(None, partial(retriever.invoke, query))
    
    if not docs:
        return ""
        
    return "\n\n".join([doc.page_content for doc in docs])


def ingest_documents() -> int:
    """
    Lê múltiplos formatos de documentos (.pdf, .md, .txt),
    realiza o split e salva no ChromaDB.
    """

    loaders = [
        DirectoryLoader(settings.rag_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(settings.rag_docs_path, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(settings.rag_docs_path, glob="**/*.txt", loader_cls=TextLoader),
    ]

    documents = []
    
    for loader in loaders:
        try:
            docs_loaded = loader.load()
            documents.extend(docs_loaded)
        except Exception as e:
            import logging
            logging.warning(f"Erro ao carregar documentos com {loader.loader_cls.__name__}: {e}")

    if not documents:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return len(chunks)