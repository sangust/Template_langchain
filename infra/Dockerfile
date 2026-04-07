# Estágio 1: Build
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=off \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.3.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

# Estágio 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copia o ambiente virtual do estágio de build
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

CMD ["python", "-m", "app.main"]