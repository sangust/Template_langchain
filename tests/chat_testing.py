"""
Testes do template LangChain.

Todos os testes usam mocks — nenhum deles precisa de Ollama ou Redis rodando.

Rode com:
    pytest tests/ -v
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Cliente HTTP do FastAPI sem dependências externas."""
    # Impede que o redis_provider tente conectar ao Redis ao ser importado
    with patch("redis.ConnectionPool"), patch("redis.Redis"):
        from app.src.api.app import app
        return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Testes de _build_messages (conversão de histórico)
# ---------------------------------------------------------------------------

class TestBuildMessages:
    """Testa a função interna que converte histórico para objetos LangChain."""

    def setup_method(self):
        from app.src.services.chat_service import _build_messages
        self.build = _build_messages

    def test_sem_historico(self):
        msgs = self.build([], "Você é um assistente.", "Olá!")
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[-1], HumanMessage)
        assert msgs[-1].content == "Olá!"

    def test_system_aparece_uma_vez(self):
        history = [
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "olá!"},
        ]
        msgs = self.build(history, "prompt", "nova pergunta")
        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1

    def test_historico_com_user_e_assistant(self):
        history = [
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        msgs = self.build(history, "prompt", "e 2+2?")
        assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "1+1?"
        assert isinstance(msgs[2], AIMessage) and msgs[2].content == "2"
        assert isinstance(msgs[3], HumanMessage) and msgs[3].content == "e 2+2?"

    def test_role_desconhecido_ignorado(self):
        history = [{"role": "system", "content": "injected"}]
        msgs = self.build(history, "prompt original", "msg")
        # Apenas o SystemMessage da função deve existir
        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "prompt original"

    def test_historico_vazio_retorna_system_e_human(self):
        msgs = self.build([], "prompt", "pergunta")
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)


# ---------------------------------------------------------------------------
# Testes de chat() — função principal
# ---------------------------------------------------------------------------

class TestChatFunction:
    """Testa chat() com o LLM mockado."""

    @patch("app.src.services.chat_service.get_llm")
    def test_retorna_answer_e_model_used(self, mock_get_llm):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="Resposta gerada")
        mock_get_llm.return_value = fake_llm

        from app.src.services.chat_service import chat
        result = chat(message="Qual é a capital do Brasil?")

        assert "answer" in result
        assert result["answer"] == "Resposta gerada"
        assert "model_used" in result

    @patch("app.src.services.chat_service.get_llm")
    def test_usa_modelo_customizado(self, mock_get_llm):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="ok")
        mock_get_llm.return_value = fake_llm

        from app.src.services.chat_service import chat
        result = chat(message="teste", model="llama3:8b")

        mock_get_llm.assert_called_once_with(model="llama3:8b")
        assert result["model_used"] == "llama3:8b"

    @patch("app.src.services.chat_service.get_llm")
    def test_historico_enviado_ao_llm(self, mock_get_llm):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="ok")
        mock_get_llm.return_value = fake_llm

        history = [
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "olá!"},
        ]

        from app.src.services.chat_service import chat
        chat(message="como vai?", history=history)

        invoke_args = fake_llm.invoke.call_args[0][0]
        # Deve haver pelo menos 4 mensagens: system + 2 histórico + user atual
        assert len(invoke_args) >= 4

    @patch("app.src.services.chat_service.get_llm")
    def test_erro_no_llm_propaga_excecao(self, mock_get_llm):
        mock_get_llm.side_effect = ConnectionError("Ollama offline")

        from app.src.services.chat_service import chat
        with pytest.raises(ConnectionError):
            chat(message="teste")


# ---------------------------------------------------------------------------
# Testes de add_message / get_history
# ---------------------------------------------------------------------------

class TestRedisHistory:
    """Testa as funções de histórico com Redis mockado."""

    @patch("app.src.services.chat_service.get_redis_client")
    def test_add_message_salva_no_redis(self, mock_get_client):
        fake_redis = MagicMock()
        mock_get_client.return_value = fake_redis

        from app.src.services.chat_service import add_message
        add_message("sessao-abc", "user", "oi")

        fake_redis.rpush.assert_called_once()
        key_usado = fake_redis.rpush.call_args[0][0]
        assert "sessao-abc" in key_usado

        payload = json.loads(fake_redis.rpush.call_args[0][1])
        assert payload["role"] == "user"
        assert payload["content"] == "oi"

    @patch("app.src.services.chat_service.get_redis_client")
    def test_add_message_faz_ltrim_e_expire(self, mock_get_client):
        fake_redis = MagicMock()
        mock_get_client.return_value = fake_redis

        from app.src.services.chat_service import add_message
        add_message("sessao-abc", "assistant", "olá!")

        fake_redis.ltrim.assert_called_once()
        fake_redis.expire.assert_called_once()

    @patch("app.src.services.chat_service.get_redis_client")
    def test_get_history_retorna_lista(self, mock_get_client):
        fake_redis = MagicMock()
        fake_redis.lrange.return_value = [
            json.dumps({"role": "user", "content": "pergunta"}),
            json.dumps({"role": "assistant", "content": "resposta"}),
        ]
        mock_get_client.return_value = fake_redis

        from app.src.services.chat_service import get_history
        history = get_history("sessao-xyz")

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @patch("app.src.services.chat_service.get_redis_client")
    def test_get_history_redis_offline_retorna_lista_vazia(self, mock_get_client):
        mock_get_client.side_effect = Exception("Redis offline")

        from app.src.services.chat_service import get_history
        history = get_history("qualquer-sessao")

        # Não deve lançar exceção — retorna lista vazia
        assert history == []

    @patch("app.src.services.chat_service.get_redis_client")
    def test_add_message_redis_offline_nao_quebra(self, mock_get_client):
        mock_get_client.side_effect = Exception("Redis offline")

        from app.src.services.chat_service import add_message
        # Não deve lançar exceção
        add_message("sessao", "user", "msg")


# ---------------------------------------------------------------------------
# Testes de endpoint HTTP
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    """Testa o endpoint POST / com dependências mockadas corretamente."""

    @patch("app.src.api.routes.home.get_session_id")
    @patch("app.src.api.routes.home.get_history")
    @patch("app.src.api.routes.home.add_message")
    @patch("app.src.api.routes.home.chat")
    def test_endpoint_retorna_200(
        self,
        mock_chat,
        mock_add_message,
        mock_get_history,
        mock_get_session_id,
    ):
        mock_get_session_id.return_value = "sessao-teste"
        mock_get_history.return_value = []

        mock_chat.return_value = {
            "answer": "Brasília",
            "model_used": "fake-model"
        }

        with patch("redis.ConnectionPool"), patch("redis.Redis"):
            from app.src.api.app import app
            c = TestClient(app)
            resp = c.post("/", json={"message": "Capital do Brasil?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Brasília"
        assert "model_used" in body

    @patch("app.src.api.routes.home.get_session_id")
    def test_endpoint_mensagem_vazia_retorna_422(self, mock_get_session_id):
        mock_get_session_id.return_value = "sessao-teste"

        with patch("redis.ConnectionPool"), patch("redis.Redis"):
            from app.src.api.app import app
            c = TestClient(app)
            resp = c.post("/", json={"message": ""})

        assert resp.status_code == 422

    @patch("app.src.api.routes.home.get_session_id")
    @patch("app.src.api.routes.home.chat")
    def test_endpoint_llm_offline_retorna_500(
        self,
        mock_chat,
        mock_get_session_id
    ):
        mock_get_session_id.return_value = "sessao-teste"
        mock_chat.side_effect = ConnectionError("Ollama offline")

        with patch("redis.ConnectionPool"), patch("redis.Redis"):
            from app.src.api.app import app
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.post("/", json={"message": "teste"})

        assert resp.status_code == 500