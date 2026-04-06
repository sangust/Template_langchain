# Template LangChain

Template reutilizável para projetos de chat com LLM local via Ollama, histórico em Redis e API FastAPI.

## Stack

- **FastAPI** — API REST + interface web
- **LangChain + Ollama** — integração com modelos locais
- **Redis** — histórico de conversa por sessão
- **Docker Compose** — orquestração de todos os serviços

## Estrutura

```
app/
├── main.py                  # entrypoint
└── src/
    ├── api/
    │   ├── app.py           # instância FastAPI
    │   └── routes/home.py   # endpoints GET e POST /
    ├── config/settings.py   # variáveis de ambiente (Pydantic Settings)
    ├── prompts/system.md    # system prompt do assistente
    ├── providers/
    │   ├── llm_provider.py  # instância cacheada do ChatOllama
    │   └── redis_provider.py# pool de conexões Redis
    ├── schemas/             # modelos Pydantic de request/response
    ├── services/
    │   ├── chat_service.py  # lógica de chat e histórico
    │   └── session_service.py
    └── rag/                 # (reservado para RAG futuro)
infra/
├── Dockerfile               # imagem da API
├── docker-compose.yml       # Ollama + Redis + API
└── .github/workflows/tests.yml
tests/
└── test_chat.py
```

## Rodando com Docker Compose

```bash
# Sobe todos os serviços
docker compose -f infra/docker-compose.yml up --build

# API disponível em http://localhost:8080
# Swagger em http://localhost:8080/docs
```

Na primeira subida o Ollama precisa baixar o modelo. Acompanhe pelo log do container `ollama`.

## Rodando localmente (sem Docker)

**Pré-requisitos:** Python 3.10+, Ollama e Redis instalados e rodando.

```bash
pip install -e .
python -m app.main
```

## Variáveis de ambiente

Crie um `.env` na raiz (ou edite `ollama.env`). Todas têm valor padrão.

| Variável | Padrão | Descrição |
|---|---|---|
| `OLLAMA_DEFAULT_MODEL` | `qwen3:4b-instruct` | Modelo a usar |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do Ollama |
| `OLLAMA_TEMPERATURE` | `0.7` | Temperatura de geração |
| `REDIS_HOST` | `localhost` | Host do Redis |
| `REDIS_PORT` | `6379` | Porta do Redis |
| `REDIS_NAMESPACE` | `historyChat` | Prefixo das chaves no Redis |
| `API_HOST` | `0.0.0.0` | Host da API |
| `API_PORT` | `8080` | Porta da API |

## Trocando o modelo

Edite `OLLAMA_DEFAULT_MODEL` no `.env`:

```
OLLAMA_DEFAULT_MODEL=llama3:8b
```

## Personalizando o system prompt

Edite `app/src/prompts/system.md`.

## Testes

```bash
pytest tests/ -v
```

Os testes não precisam de Ollama ou Redis rodando — usam mocks.