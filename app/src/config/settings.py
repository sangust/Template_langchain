from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """
    Configurações do model da IA.
    """
    vllm_base_url: str = "http://localhost:8000/v1"
    
    #models da llm
    default_model: str = "Qwen/Qwen2.5-7B-Instruct"

    #configurações do modelo e criatividade da IA (temperature: 0.0 a 1.5)
    temperature: float = 0.7
    max_tokens: int = 2048

    #configurações do prompt do sistema
    system_prompt: str = "Você é um assistente útil e educado."

    #configurações da api
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    #valida a temperatura
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.5:
            raise ValueError("temperature deve estar entre 0.0 e 1.5")
        return v

    #configurações do arquivo .env, pega automaticamente do env.
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instância global, precisa disso para inicializar as configurações uma unica vez.
settings = Settings()