import asyncio
import base64
import io
import logging
import time
import uuid

import pandas as pd
import redis as redis_lib
from langchain.tools import BaseTool

from src.connectors.dremio import client as dremio_client
from src.connectors.mysql import client as mysql_client
from src.config import REDIS_URL, EXCEL_TTL
from src.tools.utils import extract_json

logger = logging.getLogger(__name__)

_EXCEL_KEY_PREFIX = "excel:"
_LASTDF_KEY_PREFIX = "lastdf:"

_redis = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True, max_connections=10)


def store_last_df(session_id: str, df: "pd.DataFrame") -> None:
    """Guarda o último DataFrame retornado por uma tool SQL para a sessão."""
    try:
        _redis.setex(f"{_LASTDF_KEY_PREFIX}{session_id}", EXCEL_TTL, df.to_json(orient="split"))
    except Exception as e:
        logger.warning("[lastdf] Erro ao salvar último DataFrame da sessão %s: %s", session_id, e)


def df_to_excel_marker(df: "pd.DataFrame", filename: str) -> str:
    """Converte um DataFrame em arquivo Excel, salva no Redis e retorna o marcador [EXCEL:...]."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Dados")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    key = f"{_EXCEL_KEY_PREFIX}{uuid.uuid4().hex}"
    _redis.setex(key, EXCEL_TTL, b64)
    logger.info("[excel] Arquivo gerado do cache: key=%s | %d linhas | arquivo=%s", key, len(df), filename)
    return f"[EXCEL:{key}|caption:{filename}]"


def get_last_df(session_id: str) -> "pd.DataFrame | None":
    """Recupera o último DataFrame da sessão, se ainda estiver em cache."""
    try:
        data = _redis.get(f"{_LASTDF_KEY_PREFIX}{session_id}")
        if data:
            return pd.read_json(data, orient="split")
    except Exception as e:
        logger.warning("[lastdf] Erro ao recuperar DataFrame da sessão %s: %s", session_id, e)
    return None


class ExcelExportTool(BaseTool):
    name: str = "exportar_excel"
    description: str = (
        "QUANDO USAR: SOMENTE quando o usuario pedir EXPLICITAMENTE os dados em Excel, planilha ou .xlsx. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: excel, planilha, xlsx, exportar, baixar planilha, "
        "me manda em excel, retorna em excel, quero em planilha, exporta os dados. "
        "NUNCA use para respostas normais de dados em texto — so quando o usuario pedir excel/planilha explicitamente. "
        "Input: JSON com campos: "
        "'sql' (query SQL que retorna os dados desejados — pode ter quantas colunas forem necessarias, "
        "sem limite de colunas como nas outras ferramentas), "
        "'nome_arquivo' (OBRIGATORIO: nome descritivo do arquivo .xlsx com datas concretas — NUNCA use termos vagos como "
        "'ontem', 'hoje', 'semana_passada'. Exemplos: 'vendas_jan_2026.xlsx', 'vendas_03_03_a_09_03_2026.xlsx', "
        "'compras_marco_2026.xlsx'), "
        "'fonte' (OBRIGATORIO: 'dremio' para dados de vendas/faturamento/delivery/metas, "
        "'mysql' para dados de compras/fornecedores). "
        "Use a mesma logica de query SQL que usaria para as outras ferramentas, mas pode retornar "
        "todas as colunas necessarias para uma planilha completa (datas, casas, categorias, valores, etc.). "
        "Exemplo dremio: {\"sql\": \"SELECT data_evento AS data, unidade, "
        "SUM(valor_liquido) AS total FROM views.\\\"ANALYTICS\\\".\\\"fTransacoes\\\" "
        "WHERE data_evento BETWEEN '2026-01-01' AND '2026-01-31' "
        "GROUP BY 1, 2 ORDER BY data, unidade\", "
        "\"nome_arquivo\": \"vendas_dia_a_dia_jan_2026.xlsx\", \"fonte\": \"dremio\"}. "
        "Exemplo mysql: {\"sql\": \"SELECT `Fantasia`, `D. Lancamento`, `Razao Emitente`, "
        "`Descricao Item`, `V. Total` FROM `tabela_compras` WHERE ...\", "
        "\"nome_arquivo\": \"compras_marco_2026.xlsx\", \"fonte\": \"mysql\"}. "
        "Apos a ferramenta retornar, inclua na Final Answer EXATAMENTE o conteudo retornado "
        "(o marcador [EXCEL:...]) seguido de uma mensagem curta confirmando. "
        "Exemplo: '[EXCEL:excel:abc123|caption:vendas_jan_2026.xlsx]\nPlanilha gerada e enviada!'"
    )

    def _run(self, query: str) -> str:
        try:
            params = extract_json(query)
        except (ValueError, Exception) as e:
            return f"Erro: nao foi possivel interpretar o JSON da planilha. Detalhe: {e}"

        sql          = params.get("sql", "").strip()
        nome_arquivo = params.get("nome_arquivo", "dados.xlsx").strip()
        fonte        = params.get("fonte", "dremio").strip().lower()

        if not sql:
            return "Erro: campo 'sql' e obrigatorio."

        if not nome_arquivo.endswith(".xlsx"):
            nome_arquivo += ".xlsx"

        logger.info("[excel] Gerando '%s' (fonte=%s). SQL: %s", nome_arquivo, fonte, sql)
        t0 = time.time()
        try:
            df = mysql_client(sql) if fonte == "mysql" else dremio_client(sql)
        except Exception as e:
            logger.error("[excel] Erro ao executar query apos %.1fs: %s", time.time() - t0, e)
            return f"Erro ao consultar dados para o Excel: {e}"

        if df.empty:
            logger.info("[excel] Query retornou 0 linhas em %.1fs — planilha nao gerada.", time.time() - t0)
            return "Nenhum dado encontrado para gerar a planilha."

        logger.info("[excel] %d linhas obtidas em %.1fs. Gerando arquivo...", len(df), time.time() - t0)
        try:
            t1 = time.time()
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Dados")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            logger.info("[excel] Arquivo gerado em %.1fs (tamanho=%d bytes).", time.time() - t1, len(b64))
        except Exception as e:
            logger.error("[excel] Erro ao gerar arquivo: %s", e)
            return f"Erro ao gerar o arquivo Excel: {e}"

        key = f"{_EXCEL_KEY_PREFIX}{uuid.uuid4().hex}"
        _redis.setex(key, EXCEL_TTL, b64)
        logger.info("[excel] Armazenado em Redis: key=%s | %d linhas | TTL=%ds | total=%.1fs", key, len(df), EXCEL_TTL, time.time() - t0)

        return f"[EXCEL:{key}|caption:{nome_arquivo}]"

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)
