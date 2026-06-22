import logging

import redis as redis_lib

from langchain_community.chat_message_histories import RedisChatMessageHistory

from src.config import REDIS_URL

logger = logging.getLogger(__name__)

_SESSION_TTL = 864000  # 10 dias — sessões expiram após inatividade


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        ttl=_SESSION_TTL,
    )


def get_session_messages(session_id: str, since_ts: float | None = None) -> list[dict]:
    """Retorna as mensagens do histórico de uma sessão como lista de dicts {role, content}.

    Se ``since_ts`` for fornecido, retorna apenas mensagens com timestamp >= since_ts.
    Mensagens sem timestamp (gravadas antes dessa funcionalidade) são sempre incluídas.
    """
    history = get_session_history(session_id)
    result = []
    for msg in history.messages:
        role = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        ts = (msg.additional_kwargs or {}).get("timestamp")
        if since_ts is not None and ts is not None and ts < since_ts:
            continue
        result.append({"role": role, "content": msg.content})
    return result


def clear_session(session_id: str) -> None:
    """Apaga o histórico de conversa de uma sessão específica."""
    get_session_history(session_id).clear()


# Lua script: atômico — varre todas as chaves message_store:* e deleta em um único comando.
# Garante que nenhuma chave criada durante a operação seja apagada por engano.
_LUA_CLEAR_ALL = """
local keys = redis.call('KEYS', ARGV[1])
if #keys == 0 then return 0 end
return redis.call('DEL', unpack(keys))
"""


def clear_all_sessions() -> int:
    """Apaga todos os históricos de conversa de forma atômica. Retorna a quantidade de chaves removidas."""
    client = redis_lib.from_url(REDIS_URL)
    try:
        deleted = client.eval(_LUA_CLEAR_ALL, 0, "message_store:*")
        logger.info("[memory] clear_all_sessions: %d chave(s) removida(s).", deleted)
        return int(deleted)
    except Exception as e:
        logger.error("[memory] Falha ao executar clear_all_sessions via Lua: %s. Abortando sem deletar parcialmente.", e)
        raise

