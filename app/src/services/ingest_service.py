from app.src.services.rag_service import ingest_documents

if __name__ == "__main__":
    ingest_documents()
    print("Documentos ingestados com sucesso!")