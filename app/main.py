import uvicorn
from app.src.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )