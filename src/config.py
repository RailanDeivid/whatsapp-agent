import os
from dotenv import load_dotenv


load_dotenv()


def _require(name: str) -> str:
    """Lança erro claro se variável de ambiente obrigatória estiver ausente."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Variável de ambiente obrigatória não definida: {name}")
    return value


# Evolution API
EVOLUTION_API_URL                = _require("EVOLUTION_API_URL")
EVOLUTION_INSTANCE_NAME          = _require("EVOLUTION_INSTANCE_NAME")
EVOLUTION_AUTHENTICATION_API_KEY = _require("AUTHENTICATION_API_KEY")

# OpenRouter (modelo de chat)
ROUTER_API_KEY           = _require("ROUTER_API_KEY")
ROUTER_BASE_URL          = os.getenv("ROUTER_BASE_URL")
ROUTER_MODEL_NAME        = _require("ROUTER_MODEL_NAME")
ROUTER_MODEL_TEMPERATURE = float(_require("ROUTER_MODEL_TEMPERATURE"))

# Modelo com suporte a visão (usado para interpretar imagens recebidas)
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", ROUTER_MODEL_NAME)

# Whisper — OpenAI direto (apenas para transcrição de áudio)
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")

# Redis / Buffer
REDIS_URL        = _require("BOT_REDIS_URI")
BUFFER_KEY_SUFIX = os.getenv("BUFFER_KEY_SUFIX", ":buffer")
DEBOUNCE_SECONDS = float(os.getenv("DEBOUNCE_SECONDS", "3"))
BUFFER_TTL       = int(os.getenv("BUFFER_TTL", "300"))

# MySQL
DB_USER     = _require("DB_USER")
DB_PASSWORD = _require("DB_PASSWORD")
DB_HOST     = _require("DB_HOST")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_NAME     = _require("DB_NAME")

# Dremio
DREMIO_HOST         = _require("DREMIO_HOST")
DREMIO_USER         = _require("DREMIO_USER")
DREMIO_PASSWORD     = _require("DREMIO_PASSWORD")
DREMIO_POLL_INITIAL = int(os.getenv("DREMIO_POLL_INITIAL", "2"))
DREMIO_POLL_MAX     = int(os.getenv("DREMIO_POLL_MAX", "30"))

# RAG
RAG_FILES_DIR    = os.getenv("RAG_FILES_DIR", "rag_files")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vectorstore")

# Retry
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2"))

# Cache de queries
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "300"))

# Rate limiting
RATE_LIMIT_MAX    = int(os.getenv("RATE_LIMIT_MAX", "10"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Agentes
SQL_AGENT_MAX_ITERATIONS     = int(os.getenv("SQL_AGENT_MAX_ITERATIONS", "8"))
SQL_AGENT_MAX_EXECUTION_TIME = int(os.getenv("SQL_AGENT_MAX_EXECUTION_TIME", "800"))
RAG_AGENT_MAX_ITERATIONS     = int(os.getenv("RAG_AGENT_MAX_ITERATIONS", "4"))
RAG_AGENT_MAX_EXECUTION_TIME = int(os.getenv("RAG_AGENT_MAX_EXECUTION_TIME", "60"))
CONVERSATION_MAX_HISTORY     = int(os.getenv("CONVERSATION_MAX_HISTORY", "5"))

# MySQL
MYSQL_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "5"))

# Dremio — limite de segurança para resultados (evita explosão de memória)
DREMIO_MAX_ROWS = int(os.getenv("DREMIO_MAX_ROWS", "50000"))

# Dremio — máximo de queries simultâneas (evita fila no Dremio)
DREMIO_MAX_CONCURRENT = int(os.getenv("DREMIO_MAX_CONCURRENT", "3"))

# Excel — TTL do arquivo no Redis antes de expirar (segundos)
EXCEL_TTL = int(os.getenv("EXCEL_TTL", "300"))

# Modelo de fallback (usado se o principal falhar por rate limit ou erro de API)
OPENAI_FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_NAME", "")

# Controle de acesso
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/access.db")

# Formato: TELEFONE:NOME:SETOR:CASA:admin|user (separados por vírgula)
def _parse_seed_users(raw: str) -> list[dict]:
    import logging
    _log = logging.getLogger(__name__)
    users = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) < 4:
            _log.warning("SEED_USERS: entrada ignorada por formato invalido (esperado TELEFONE:NOME:CARGO:CASA): %r", entry)
            continue
        users.append({
            "telefone": parts[0],
            "nome":     parts[1],
            "cargo":    parts[2],
            "casa":     parts[3],
            "is_admin": 1 if len(parts) >= 5 and parts[4].lower() == "admin" else 0,
        })
    return users

SEED_USERS: list[dict] = _parse_seed_users(os.getenv("SEED_USERS", ""))

# Contato de suporte exibido na mensagem de acesso negado
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "um administrador")
