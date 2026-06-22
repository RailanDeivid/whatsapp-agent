import hashlib
import io
import logging
import threading
import time
from typing import TypeVar, Callable

import pandas as pd
import redis as redis_lib
import requests

from src.config import DREMIO_HOST, DREMIO_USER, DREMIO_PASSWORD, RETRY_MAX_ATTEMPTS, RETRY_BACKOFF_BASE, DREMIO_POLL_INITIAL, DREMIO_POLL_MAX, DREMIO_MAX_ROWS, REDIS_URL, QUERY_CACHE_TTL

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_redis = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True, max_connections=20)
_QCACHE_PREFIX = "qcache:"


def _qcache_key(sql: str) -> str:
    return _QCACHE_PREFIX + hashlib.md5(sql.strip().lower().encode()).hexdigest()


def _cache_get(sql: str) -> pd.DataFrame | None:
    try:
        data = _redis.get(_qcache_key(sql))
        if data:
            logger.info("[dremio-cache] HIT para query: %.80s", sql)
            return pd.read_json(io.StringIO(data), orient="split")
    except Exception as e:
        logger.warning("[dremio-cache] Erro ao ler cache: %s", e)
    return None


def _cache_set(sql: str, df: pd.DataFrame) -> None:
    try:
        _redis.setex(_qcache_key(sql), QUERY_CACHE_TTL, df.to_json(orient="split"))
        logger.info("[dremio-cache] Resultado cacheado (TTL=%ds, %d linhas): %.80s", QUERY_CACHE_TTL, len(df), sql)
    except Exception as e:
        logger.warning("[dremio-cache] Erro ao gravar cache: %s", e)


def _with_retry(fn: Callable[[], _T], label: str) -> _T:
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == RETRY_MAX_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("%s falhou (tentativa %d/%d): %s — retry em %.0fs", label, attempt, RETRY_MAX_ATTEMPTS, exc, wait)
            time.sleep(wait)

_token_cache: dict[str, tuple[str, float]] = {}
_token_lock = threading.Lock()
_TOKEN_TTL = 3300  # tokens Dremio duram ~1h — usa 55min para renovar antes de expirar


def _get_token(host: str, username: str, password: str) -> str:
    cache_key = f"{host}:{username}"
    with _token_lock:
        cached = _token_cache.get(cache_key)
        if cached:
            token, created_at = cached
            if time.time() - created_at < _TOKEN_TTL:
                age = int(time.time() - created_at)
                logger.debug("Token Dremio obtido do cache (age=%ds)", age)
                return token
            logger.info("Token Dremio expirado — renovando...")

        logger.info("Autenticando no Dremio (host=%s, user=%s)...", host, username)

        def _do_login():
            res = requests.post(
                f"http://{host}/apiv2/login",
                json={"userName": username, "password": password},
                timeout=10,
            )
            res.raise_for_status()
            return res.json()["token"]

        token = _with_retry(_do_login, "Login Dremio")
        _token_cache[cache_key] = (token, time.time())
        logger.info("Token Dremio obtido com sucesso.")
        return token


def client(sql: str, max_wait: int = 360) -> pd.DataFrame:
    """Executa query SQL no Dremio e retorna DataFrame (com cache Redis).

    max_wait: tempo máximo em segundos aguardando o job concluir (padrão 360s).
    """
    cached = _cache_get(sql)
    if cached is not None:
        return cached

    t_total = time.time()
    logger.debug("Token Dremio sendo obtido...")
    token = _get_token(DREMIO_HOST, DREMIO_USER, DREMIO_PASSWORD)
    headers = {"Authorization": f"_dremio{token}"}
    base = f"http://{DREMIO_HOST}"

    logger.info("Submetendo query ao Dremio: %s", sql)

    def _do_submit():
        res = requests.post(f"{base}/api/v3/sql", headers=headers, json={"sql": sql}, timeout=30)
        res.raise_for_status()
        return res.json()["id"]

    job_id = _with_retry(_do_submit, "Submit query Dremio")
    logger.info("Job criado: %s. Aguardando conclusao...", job_id)
    t_job_start = time.time()

    t_start = time.time()
    last_state = None
    poll_interval = DREMIO_POLL_INITIAL

    while time.time() - t_start < max_wait:
        try:
            status_res = requests.get(
                f"{base}/api/v3/job/{job_id}", headers=headers, timeout=30
            )
            job_state = status_res.json().get("jobState")
        except requests.exceptions.ConnectTimeout:
            elapsed = int(time.time() - t_start)
            logger.warning("[dremio] ConnectTimeout ao checar status do job %s (%ds decorridos) — poll_interval=%.0fs. Tentando novamente...", job_id, elapsed, poll_interval)
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, DREMIO_POLL_MAX)
            continue
        except Exception as e:
            elapsed = int(time.time() - t_start)
            logger.warning("[dremio] Erro ao checar status do job %s (%ds): %s — tentando novamente...", job_id, elapsed, e)
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, DREMIO_POLL_MAX)
            continue

        if job_state != last_state:
            logger.info("[dremio] Job %s: estado %s (%ds decorridos)", job_id, job_state, int(time.time() - t_start))
            last_state = job_state

        if job_state == "COMPLETED":
            logger.info("[dremio] Job %s concluido em %.1fs.", job_id, time.time() - t_job_start)
            break
        if job_state in ("FAILED", "CANCELED"):
            try:
                error = status_res.json().get("errorMessage", "Erro desconhecido")
            except Exception:
                error = "Erro desconhecido (resposta nao e JSON)"
            logger.error("[dremio] Job %s falhou com estado '%s' apos %.1fs: %s", job_id, job_state, time.time() - t_job_start, error)
            raise RuntimeError(f"Job Dremio falhou ({job_state}): {error}")

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 2, DREMIO_POLL_MAX)
    else:
        logger.error("[dremio] Timeout de %ds atingido aguardando job %s (ultimo estado: %s). Verifique reflections/performance da view.", max_wait, job_id, last_state)
        raise TimeoutError(f"Timeout aguardando job Dremio ({max_wait}s) — job={job_id}, ultimo_estado={last_state}")

    all_rows: list = []
    columns: list = []
    offset = 0
    limit = 500
    first_page = True

    while True:
        result_res = requests.get(
            f"{base}/api/v3/job/{job_id}/results?offset={offset}&limit={limit}",
            headers=headers,
            timeout=30,
        )
        result_res.raise_for_status()
        data = result_res.json()

        if first_page:
            columns = [col["name"] for col in data.get("schema", [])]
            first_page = False

        rows = data.get("rows", [])
        if not rows:
            break

        all_rows.extend(rows)
        offset += limit
        logger.debug("Dremio paginacao: %d linhas carregadas (offset=%d)", len(all_rows), offset)

        if len(all_rows) >= DREMIO_MAX_ROWS:
            logger.warning("Dremio: limite de %d linhas atingido — truncando resultado (job=%s).", DREMIO_MAX_ROWS, job_id)
            break

    elapsed_total = time.time() - t_total
    logger.info("Dremio: %d linhas retornadas em %.1fs (job=%s)", len(all_rows), elapsed_total, job_id)
    df = pd.DataFrame(all_rows, columns=columns) if all_rows else pd.DataFrame(columns=columns)
    if not df.empty:
        _cache_set(sql, df)
    return df
