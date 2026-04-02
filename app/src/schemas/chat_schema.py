from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    Schema para requisição de chat por parte do cliente para a LLM.
    """
    # Mensagem do usuário
    message: str = Field(..., description="Mensagem do usuário", min_length=1)
    
    # Exemplo de uso para documentação da API no Swagger
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Explique o que é uma API REST",
                    "system_prompt": "Você é um professor de programação.",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """
    Schema para resposta de chat por parte da LLM para o cliente.
    """
    answer: str = Field(..., description="Resposta gerada pelo modelo")
    model_used: str = Field(..., description="Modelo que gerou a resposta")

