import hashlib
import io
import logging
import threading
import time

import pandas as pd
import redis as redis_lib
from mysql.connector.pooling import MySQLConnectionPool

from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, RETRY_MAX_ATTEMPTS, RETRY_BACKOFF_BASE, MYSQL_POOL_SIZE, REDIS_URL, QUERY_CACHE_TTL

logger = logging.getLogger(__name__)

_redis = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True, max_connections=20)
_QCACHE_PREFIX = "qcache:"


def _qcache_key(sql: str) -> str:
    return _QCACHE_PREFIX + hashlib.md5(sql.strip().lower().encode()).hexdigest()


def _cache_get(sql: str) -> pd.DataFrame | None:
    try:
        data = _redis.get(_qcache_key(sql))
        if data:
            logger.info("[mysql-cache] HIT para query: %.80s", sql)
            return pd.read_json(io.StringIO(data), orient="split")
    except Exception as e:
        logger.warning("[mysql-cache] Erro ao ler cache: %s", e)
    return None


def _cache_set(sql: str, df: pd.DataFrame) -> None:
    try:
        _redis.setex(_qcache_key(sql), QUERY_CACHE_TTL, df.to_json(orient="split"))
        logger.info("[mysql-cache] Resultado cacheado (TTL=%ds, %d linhas): %.80s", QUERY_CACHE_TTL, len(df), sql)
    except Exception as e:
        logger.warning("[mysql-cache] Erro ao gravar cache: %s", e)


_pool: MySQLConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> MySQLConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                logger.info("Criando pool de conexões MySQL...")
                _pool = MySQLConnectionPool(
                    pool_name="bot_pool",
                    pool_size=MYSQL_POOL_SIZE,
                    host=DB_HOST,
                    port=DB_PORT,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_NAME,
                    use_pure=True,
                )
    return _pool


def client(sql: str) -> pd.DataFrame:
    """
    Executa uma query SQL no MySQL e retorna um DataFrame (com cache Redis).
    Usa connection pooling e retry com backoff exponencial.
    """
    cached = _cache_get(sql)
    if cached is not None:
        return cached

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        db = _get_pool().get_connection()
        cursor = None
        try:
            cursor = db.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            df = pd.DataFrame(rows, columns=columns)
            if not df.empty:
                _cache_set(sql, df)
            return df
        except Exception as exc:
            errno = getattr(exc, "errno", None)
            # Erros permanentes: sintaxe (1064), acesso negado (1045), tabela inexistente (1146),
            # banco inexistente (1049), coluna inexistente (1054), permissao negada (1142)
            _PERMANENT_ERRORS = (1064, 1045, 1146, 1049, 1054, 1142)
            if errno in _PERMANENT_ERRORS:
                logger.error("[mysql] Erro permanente (errno=%s) — sem retry. Query: %.100s | Erro: %s", errno, sql, exc)
                raise
            if attempt == RETRY_MAX_ATTEMPTS:
                logger.error("[mysql] Falha apos %d tentativas. Query: %.100s | Ultimo erro: %s", RETRY_MAX_ATTEMPTS, sql, exc)
                raise
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("[mysql] Falha temporaria (tentativa %d/%d, errno=%s): %s — retry em %.0fs", attempt, RETRY_MAX_ATTEMPTS, errno, exc, wait)
            time.sleep(wait)
        finally:
            if cursor is not None:
                cursor.close()
            db.close()
