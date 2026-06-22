"""
Configura variáveis de ambiente falsas e mocks de dependências pesadas
antes de qualquer import de src/. Necessário para rodar testes fora do Docker.
"""
import os
import sys
from unittest.mock import MagicMock

# ── Mocks estruturados (precisam de classes reais para herança funcionar) ──────

class _FakeBaseTool:
    """Substituto mínimo de langchain BaseTool para que as tools possam ser instanciadas."""
    name: str = ""
    description: str = ""
    def _run(self, query: str) -> str:
        return ""
    async def _arun(self, query: str) -> str:
        return ""

class _FakeRedisChatMessageHistory:
    def __init__(self, **kwargs):
        self.messages = []
    def clear(self):
        self.messages = []
    def add_user_message(self, msg):
        pass
    def add_ai_message(self, msg):
        pass

_fake_langchain_tools = MagicMock()
_fake_langchain_tools.BaseTool = _FakeBaseTool
sys.modules["langchain.tools"] = _fake_langchain_tools

_fake_history = MagicMock()
_fake_history.RedisChatMessageHistory = _FakeRedisChatMessageHistory
sys.modules["langchain_community.chat_message_histories"] = _fake_history

# ── Mocks genéricos (MagicMock é suficiente) ───────────────────────────────────
for _mod in [
    # Redis
    "redis",
    "redis.asyncio",
    "redis.asyncio.client",
    # LangChain
    "langchain",
    "langchain.agents",
    "langchain.prompts",
    "langchain_core",
    "langchain_core.messages",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.document_loaders",
    "langchain_chroma",
    "langchain_openai",
    "langchain_text_splitters",
    "langchain_text_splitters.character",
    # FastAPI (pydantic não é mockado — está instalado no Anaconda)
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    # Vector / Chroma
    "chromadb",
    # Plotting
    "seaborn",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "matplotlib.patches",
    "matplotlib.figure",
    "matplotlib.axes",
    # Data / ML
    "mysql",
    "mysql.connector",
    "mysql.connector.pooling",
    "openai",
]:
    sys.modules.setdefault(_mod, MagicMock())

_FAKE_ENV = {
    "EVOLUTION_API_URL": "http://fake-evolution",
    "EVOLUTION_INSTANCE_NAME": "fake-instance",
    "AUTHENTICATION_API_KEY": "fake-api-key",
    "ROUTER_API_KEY": "fake-openai-key",
    "ROUTER_MODEL_NAME": "x-ai/grok-4.1-fast",
    "OPENAI_MODEL_TEMPERATURE": "0",
    "BOT_REDIS_URI": "redis://localhost:6379",
    "DB_USER": "fake",
    "DB_PASSWORD": "fake",
    "DB_HOST": "localhost",
    "DB_NAME": "fake_db",
    "DREMIO_HOST": "localhost",
    "DREMIO_USER": "fake",
    "DREMIO_PASSWORD": "fake",
}

for _k, _v in _FAKE_ENV.items():
    os.environ.setdefault(_k, _v)

import pytest

@pytest.fixture
def anyio_backend():
    return "asyncio"
