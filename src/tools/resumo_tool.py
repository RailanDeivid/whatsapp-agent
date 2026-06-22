import asyncio
import concurrent.futures
import json
import logging
import re
import threading
from datetime import datetime

from langchain.tools import BaseTool

from src.connectors.dremio import client
from src.tools.utils import fmt_brl as _fmt_brl, fmt_int_br as _fmt_int_br

logger = logging.getLogger(__name__)


def _validate_date(d: str) -> bool:
    """Valida que a string está no formato AAAA-MM-DD e é uma data real."""
    try:
        datetime.strptime(d, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _safe_float(val) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


# ─── WHERE builder ────────────────────────────────────────────────────────────

def _build_where(
    periodo_inicio: str,
    periodo_fim: str,
    filtro_casa: str,
    filtro_alavanca: str,
    filtro_marca: str,
    date_col: str = "data_evento",
) -> str:
    clause = f"{date_col} BETWEEN '{periodo_inicio}' AND '{periodo_fim}'"
    if filtro_casa:
        clause += f" AND unidade = '{filtro_casa}'"
    elif filtro_alavanca:
        clause += f" AND segmento = '{filtro_alavanca}'"
    elif filtro_marca:
        clause += f" AND marca = '{filtro_marca}'"
    return clause


# ─── Queries individuais (executadas em paralelo) ─────────────────────────────

def _query_vendas(where: str) -> str:
    # Query simples sem window function — pct calculado no Python
    sql = f"""
SELECT
    Grande_Grupo,
    SUM(valor_liquido) AS total_grupo,
    SUM(qtd_pessoas) AS fluxo_grupo
FROM views."ANALYTICS"."fTransacoes"
WHERE {where}
GROUP BY Grande_Grupo
ORDER BY total_grupo DESC
"""
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return "*Vendas*\n- Sem dados no período."

        fat = _safe_float(df["total_grupo"].sum())
        fluxo = _safe_float(df["fluxo_grupo"].sum())
        tm = fat / fluxo if fluxo else 0.0

        lines = [
            "*Vendas*",
            f"- Faturamento: {_fmt_brl(fat)}",
            f"- Ticket médio: {_fmt_brl(tm)}",
            f"- Fluxo de pessoas: {_fmt_int_br(int(fluxo))} pax",
            "",
            "*Mix de vendas*",
        ]
        for _, row in df.iterrows():
            grupo = row["Grande_Grupo"] or "OUTROS"
            total = _safe_float(row["total_grupo"])
            pct = (total / fat * 100) if fat else 0.0
            pct_str = f"{pct:.2f}".replace(".", ",")
            lines.append(f"- {grupo}: {_fmt_brl(total)} ({pct_str}%)")

        return "\n".join(lines)

    except Exception as e:
        logger.error("[resumo] Erro em _query_vendas: %s", e)
        return "*Vendas*\n- Dados indisponíveis no período."


def _query_delivery(where: str) -> str:
    sql = f"""
SELECT
    nome_funcionario AS plataforma,
    SUM(valor_liquido) AS total
FROM views."ANALYTICS"."fDelivery"
WHERE {where}
GROUP BY nome_funcionario
ORDER BY total DESC
"""
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return "*Delivery*\n- Sem dados no período."

        lines = ["*Delivery*"]
        grand_total = 0.0
        for _, row in df.iterrows():
            plat = row["plataforma"] or "OUTROS"
            val = _safe_float(row["total"])
            grand_total += val
            lines.append(f"- {plat}: {_fmt_brl(val)}")
        lines.append(f"- Total delivery: {_fmt_brl(grand_total)}")
        return "\n".join(lines)

    except Exception as e:
        logger.error("[resumo] Erro em _query_delivery: %s", e)
        return "*Delivery*\n- Dados indisponíveis no período."


def _query_estornos(where: str) -> str:
    sql = f"""
SELECT
    SUM(valor_produto) AS total_estornos,
    SUM(quantidade)    AS qtd_estornos
FROM views."ANALYTICS"."fEstornos"
WHERE {where}
"""
    try:
        df = client(sql, max_wait=600)
        if df.empty or df["total_estornos"].iloc[0] is None:
            return "*Estornos*\n- Sem dados no período."

        total = _safe_float(df["total_estornos"].iloc[0])
        qtd = _safe_int(df["qtd_estornos"].iloc[0])
        return f"*Estornos*\n- Total estornado: {_fmt_brl(total)} ({_fmt_int_br(qtd)} itens)"

    except Exception as e:
        logger.error("[resumo] Erro em _query_estornos: %s", e)
        return "*Estornos*\n- Dados indisponíveis no período."


def _query_cortesias(where: str) -> str:
    sql = f"""
SELECT
    ROUND(SUM(total_cortesias), 0) AS total_cortesias
FROM views."ANALYTICS"."fCortesias"
WHERE {where}
"""
    try:
        df = client(sql, max_wait=600)
        if df.empty or df["total_cortesias"].iloc[0] is None:
            return "*Cortesias*\n- Sem dados no período."

        total = _safe_float(df["total_cortesias"].iloc[0])
        return f"*Cortesias*\n- Total cortesias: {_fmt_brl(total)}"

    except Exception as e:
        logger.error("[resumo] Erro em _query_cortesias: %s", e)
        return "*Cortesias*\n- Dados indisponíveis no período."


def _query_pagamentos(where_pagto: str) -> str:
    sql = f"""
SELECT
    descricao_corretas AS forma,
    SUM(vl_recebido) AS total
FROM views."ANALYTICS"."fFormasPagamento"
WHERE {where_pagto}
GROUP BY descricao_corretas
ORDER BY total DESC
"""
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return "*Formas de pagamento*\n- Sem dados no período."

        lines = ["*Formas de pagamento*"]
        grand_total = 0.0
        for _, row in df.iterrows():
            forma = row["forma"] or "OUTROS"
            val = _safe_float(row["total"])
            grand_total += val
            lines.append(f"- {forma}: {_fmt_brl(val)}")
        lines.append(f"- Total: {_fmt_brl(grand_total)}")
        return "\n".join(lines)

    except Exception as e:
        logger.error("[resumo] Erro em _query_pagamentos: %s", e)
        return "*Formas de pagamento*\n- Dados indisponíveis no período."


# ─── Queries POR CASA (modo casa a casa quando o escopo é vertical/marca) ─────

def _query_vendas_por_casa(where: str):
    """Retorna (blocos_por_casa, totais_faturamento_por_casa)."""
    sql = f"""
SELECT
    unidade,
    Grande_Grupo,
    SUM(valor_liquido) AS total_grupo,
    SUM(qtd_pessoas) AS fluxo_grupo
FROM views."ANALYTICS"."fTransacoes"
WHERE {where}
GROUP BY unidade, Grande_Grupo
"""
    blocos: dict = {}
    totais: dict = {}
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return blocos, totais

        for casa, gdf in df.groupby("unidade"):
            if not casa:
                continue
            fat = _safe_float(gdf["total_grupo"].sum())
            fluxo = _safe_float(gdf["fluxo_grupo"].sum())
            tm = fat / fluxo if fluxo else 0.0
            totais[casa] = fat

            lines = [
                "*Vendas*",
                f"- Faturamento: {_fmt_brl(fat)}",
                f"- Ticket médio: {_fmt_brl(tm)}",
                f"- Fluxo de pessoas: {_fmt_int_br(int(fluxo))} pax",
                "",
                "*Mix de vendas*",
            ]
            for _, row in gdf.sort_values("total_grupo", ascending=False).iterrows():
                grupo = row["Grande_Grupo"] or "OUTROS"
                total = _safe_float(row["total_grupo"])
                pct = (total / fat * 100) if fat else 0.0
                pct_str = f"{pct:.2f}".replace(".", ",")
                lines.append(f"- {grupo}: {_fmt_brl(total)} ({pct_str}%)")
            blocos[casa] = "\n".join(lines)

        return blocos, totais

    except Exception as e:
        logger.error("[resumo] Erro em _query_vendas_por_casa: %s", e)
        return blocos, totais


def _query_delivery_por_casa(where: str) -> dict:
    sql = f"""
SELECT
    unidade,
    nome_funcionario AS plataforma,
    SUM(valor_liquido) AS total
FROM views."ANALYTICS"."fDelivery"
WHERE {where}
GROUP BY unidade, nome_funcionario
"""
    blocos: dict = {}
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return blocos
        for casa, gdf in df.groupby("unidade"):
            if not casa:
                continue
            lines = ["*Delivery*"]
            grand_total = 0.0
            for _, row in gdf.sort_values("total", ascending=False).iterrows():
                plat = row["plataforma"] or "OUTROS"
                val = _safe_float(row["total"])
                grand_total += val
                lines.append(f"- {plat}: {_fmt_brl(val)}")
            lines.append(f"- Total delivery: {_fmt_brl(grand_total)}")
            blocos[casa] = "\n".join(lines)
        return blocos
    except Exception as e:
        logger.error("[resumo] Erro em _query_delivery_por_casa: %s", e)
        return blocos


def _query_estornos_por_casa(where: str) -> dict:
    sql = f"""
SELECT
    unidade,
    SUM(valor_produto) AS total_estornos,
    SUM(quantidade)    AS qtd_estornos
FROM views."ANALYTICS"."fEstornos"
WHERE {where}
GROUP BY unidade
"""
    blocos: dict = {}
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return blocos
        for _, row in df.iterrows():
            casa = row["unidade"]
            if not casa:
                continue
            total = _safe_float(row["total_estornos"])
            qtd = _safe_int(row["qtd_estornos"])
            blocos[casa] = f"*Estornos*\n- Total estornado: {_fmt_brl(total)} ({_fmt_int_br(qtd)} itens)"
        return blocos
    except Exception as e:
        logger.error("[resumo] Erro em _query_estornos_por_casa: %s", e)
        return blocos


def _query_cortesias_por_casa(where: str) -> dict:
    sql = f"""
SELECT
    unidade,
    ROUND(SUM(total_cortesias), 0) AS total_cortesias
FROM views."ANALYTICS"."fCortesias"
WHERE {where}
GROUP BY unidade
"""
    blocos: dict = {}
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return blocos
        for _, row in df.iterrows():
            casa = row["unidade"]
            if not casa:
                continue
            total = _safe_float(row["total_cortesias"])
            blocos[casa] = f"*Cortesias*\n- Total cortesias: {_fmt_brl(total)}"
        return blocos
    except Exception as e:
        logger.error("[resumo] Erro em _query_cortesias_por_casa: %s", e)
        return blocos


def _query_pagamentos_por_casa(where_pagto: str) -> dict:
    sql = f"""
SELECT
    unidade,
    descricao_corretas AS forma,
    SUM(vl_recebido) AS total
FROM views."ANALYTICS"."fFormasPagamento"
WHERE {where_pagto}
GROUP BY unidade, descricao_corretas
"""
    blocos: dict = {}
    try:
        df = client(sql, max_wait=600)
        if df.empty:
            return blocos
        for casa, gdf in df.groupby("unidade"):
            if not casa:
                continue
            lines = ["*Formas de pagamento*"]
            grand_total = 0.0
            for _, row in gdf.sort_values("total", ascending=False).iterrows():
                forma = row["forma"] or "OUTROS"
                val = _safe_float(row["total"])
                grand_total += val
                lines.append(f"- {forma}: {_fmt_brl(val)}")
            lines.append(f"- Total: {_fmt_brl(grand_total)}")
            blocos[casa] = "\n".join(lines)
        return blocos
    except Exception as e:
        logger.error("[resumo] Erro em _query_pagamentos_por_casa: %s", e)
        return blocos


# ─── Insight LLM (chamada separada com max_tokens curto) ──────────────────────
# Modelo dedicado para gerar 1-2 linhas de insight a partir do resumo formatado.
# Usado apos a tool retornar o output completo (return_direct=True bypassa o LLM
# do agente, entao geramos o insight aqui, com max_tokens fixo curto pra nunca truncar).

_insight_model = None
_insight_model_lock = threading.Lock()


def _get_insight_model():
    global _insight_model
    if _insight_model is None:
        with _insight_model_lock:
            if _insight_model is None:
                from langchain_openai import ChatOpenAI
                from src.config import ROUTER_API_KEY, ROUTER_BASE_URL, ROUTER_MODEL_NAME, OPENAI_FALLBACK_MODEL
                _insight_model = ChatOpenAI(
                    model=OPENAI_FALLBACK_MODEL or ROUTER_MODEL_NAME,
                    temperature=0.3,
                    base_url=ROUTER_BASE_URL,
                    api_key=ROUTER_API_KEY,
                    max_tokens=250,
                )
    return _insight_model


def _generate_insight(resumo_text: str) -> str:
    """Gera ate 2 linhas de insight a partir do resumo formatado.

    Retorna string vazia em caso de falha — o resumo segue sem insight.
    """
    if not resumo_text or len(resumo_text) < 200:
        return ""

    prompt = (
        "Voce e analista de dados de uma rede de unidades de negocio. "
        "Analise o resumo abaixo e gere ATE 2 linhas curtas de insight direto sobre o que se "
        "destaca: lider, anomalia, concentracao, disparidade, padrao incomum ou diferenca de mix. "
        "NUNCA comece com 'observa-se', 'destaca-se', 'nota-se', 'olhe que', 'veja' ou similar — va direto ao fato. "
        "Use proporcoes/percentuais quando fizer sentido em vez de valores absolutos. "
        "Formate envolvendo TODO o texto com underscore _ no inicio e fim (italico WhatsApp). "
        "Exemplo de formato: \"_BPIPA lidera com 25% do total da vertical. Cortesias do BPI chamam atencao — 27% acima da media._\""
        f"\n\n--- RESUMO ---\n{resumo_text}"
    )

    try:
        model = _get_insight_model()
        result = model.invoke(prompt)
        insight = (result.content or "").strip()
        if not insight:
            return ""
        if not insight.startswith("_"):
            insight = "_" + insight
        if not insight.endswith("_"):
            insight = insight + "_"
        return insight
    except Exception as e:
        logger.warning("[resumo] Falha ao gerar insight: %s", e)
        return ""


# ─── Parser de input (JSON ou Python kwargs) ──────────────────────────────────

def _parse_input(text: str) -> dict:
    """Aceita JSON ou Python kwargs (key='value'). Retorna dict com os campos."""
    text = text.strip()
    # Tenta JSON primeiro
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: key='value' ou key="value" ou key=value
    result = {}
    for m in re.finditer(r"(\w+)\s*=\s*['\"]?([^,'\"\n]+)['\"]?", text):
        result[m.group(1).strip()] = m.group(2).strip()
    return result


# ─── Tool ─────────────────────────────────────────────────────────────────────

class DremioResumoTool(BaseTool):
    name: str = "gerar_resumo"
    # Resumos com escopo de vertical podem retornar 30KB+ — passar pelo LLM
    # gerar Final Answer trunca por limite de tokens. O output da tool ja vem
    # formatado pronto pra envio, entao vira Final Answer direto.
    return_direct: bool = True
    description: str = (
        "Gera resumo consolidado de vendas consultando em PARALELO: "
        "faturamento, ticket médio, fluxo de pessoas, mix de produtos por categoria, "
        "delivery por plataforma, total de estornos e total de cortesias. "
        "Use UNICAMENTE quando a mensagem do usuario contiver uma destas tres palavras exatas: "
        "'resumo', 'visao geral' ou 'panorama'. "
        "Se a mensagem NAO contiver nenhuma dessas tres palavras, NUNCA use esta ferramenta — "
        "use consultar_transacoes ou a ferramenta especifica adequada. "
        "Exemplos que NAO ativam: 'tras as vendas', 'faturamento de', 'dados de', 'quanto vendeu', "
        "'como foi', 'tras do X', 'vendas de hoje', 'mostra as vendas'. "
        "Exemplos que ATIVAM: 'resumo do bar', 'visao geral da semana', 'panorama de marco'. "
        "Input: JSON com os campos: "
        "periodo_inicio (AAAA-MM-DD, obrigatorio), periodo_fim (AAAA-MM-DD, obrigatorio), "
        "filtro_casa (nome exato da casa, opcional), "
        "filtro_alavanca (Tipo_A ou Tipo_B ou Tipo_C, opcional), "
        "filtro_marca (marca/bandeira, opcional), "
        "por_casa (true ou false, opcional — quando true e o escopo for segmento/marca, retorna um bloco completo por casa ordenado pelo faturamento; quando false, retorna um unico total consolidado do escopo). "
        "Passe APENAS UM dos tres filtros (filtro_casa OU filtro_alavanca OU filtro_marca). "
        'Exemplo casa unica: {"periodo_inicio": "2026-04-13", "periodo_fim": "2026-04-19", "filtro_casa": "TBI"} '
        'Exemplo vertical casa a casa: {"periodo_inicio": "2026-04-13", "periodo_fim": "2026-04-19", "filtro_alavanca": "Tipo_A", "por_casa": true} '
        'Exemplo vertical consolidada: {"periodo_inicio": "2026-04-13", "periodo_fim": "2026-04-19", "filtro_alavanca": "Tipo_A", "por_casa": false}'
    )

    def _run(self, query: str) -> str:
        params = _parse_input(query)
        periodo_inicio = params.get("periodo_inicio", "")
        periodo_fim = params.get("periodo_fim", "")
        filtro_casa = params.get("filtro_casa", "")
        filtro_alavanca = params.get("filtro_alavanca", "")
        filtro_marca = params.get("filtro_marca", "")
        raw_por_casa = params.get("por_casa", None)

        if not periodo_inicio or not periodo_fim:
            return "Erro: periodo_inicio e periodo_fim sao obrigatorios no formato AAAA-MM-DD."

        if not _validate_date(periodo_inicio):
            logger.warning("[resumo] Data de inicio invalida recebida: %r", periodo_inicio)
            return f"Erro: periodo_inicio '{periodo_inicio}' nao e uma data valida. Use o formato AAAA-MM-DD (ex: 2026-04-13)."
        if not _validate_date(periodo_fim):
            logger.warning("[resumo] Data de fim invalida recebida: %r", periodo_fim)
            return f"Erro: periodo_fim '{periodo_fim}' nao e uma data valida. Use o formato AAAA-MM-DD (ex: 2026-04-19)."
        if periodo_inicio > periodo_fim:
            logger.warning("[resumo] periodo_inicio (%s) e posterior a periodo_fim (%s).", periodo_inicio, periodo_fim)
            return f"Erro: periodo_inicio ({periodo_inicio}) nao pode ser posterior a periodo_fim ({periodo_fim})."

        if isinstance(raw_por_casa, bool):
            por_casa = raw_por_casa
        elif raw_por_casa is None:
            por_casa = bool(filtro_alavanca) and not filtro_casa
        else:
            por_casa = str(raw_por_casa).strip().lower() in ("true", "1", "yes", "sim")
        if filtro_casa:
            por_casa = False

        where = _build_where(periodo_inicio, periodo_fim, filtro_casa, filtro_alavanca, filtro_marca)
        where_pagto = _build_where(periodo_inicio, periodo_fim, filtro_casa, filtro_alavanca, filtro_marca, date_col="data")
        escopo = filtro_casa or filtro_alavanca or filtro_marca or "Geral"
        di = f"{periodo_inicio[8:10]}/{periodo_inicio[5:7]}/{periodo_inicio[:4]}"
        df_ = f"{periodo_fim[8:10]}/{periodo_fim[5:7]}/{periodo_fim[:4]}"
        header = f"*Resumo — {escopo} | {di} a {df_}*"

        logger.info(
            "[resumo] Disparando queries em paralelo para escopo=%s | %s a %s | por_casa=%s",
            escopo, periodo_inicio, periodo_fim, por_casa,
        )

        if por_casa:
            result = self._run_por_casa(header, where, where_pagto)
            insight = _generate_insight(result)
            return f"{result}\n\n{insight}" if insight else result

        _FALLBACKS = {
            "vendas":     "*Vendas*\n- Dados indisponíveis no período.",
            "delivery":   "*Delivery*\n- Dados indisponíveis no período.",
            "pagamentos": "*Formas de pagamento*\n- Dados indisponíveis no período.",
            "estornos":   "*Estornos*\n- Dados indisponíveis no período.",
            "cortesias":  "*Cortesias*\n- Dados indisponíveis no período.",
        }

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        fut_map = {
            "vendas":     pool.submit(_query_vendas, where),
            "delivery":   pool.submit(_query_delivery, where),
            "pagamentos": pool.submit(_query_pagamentos, where_pagto),
            "estornos":   pool.submit(_query_estornos, where),
            "cortesias":  pool.submit(_query_cortesias, where),
        }

        # Aguarda até 580s pelo conjunto (abaixo do max_wait=600s passado ao Dremio).
        # Queries que não terminarem são descartadas — não bloqueamos pelo mais lento.
        concurrent.futures.wait(fut_map.values(), timeout=580)
        pool.shutdown(wait=False)

        def _collect(fut: concurrent.futures.Future, section: str) -> str:
            if fut.done():
                try:
                    return fut.result()
                except Exception as e:
                    logger.warning("[resumo] Falha em %s: %s", section, e)
            else:
                logger.warning("[resumo] Timeout em %s — descartando resultado.", section)
            return _FALLBACKS[section]

        vendas     = _collect(fut_map["vendas"],     "vendas")
        delivery   = _collect(fut_map["delivery"],   "delivery")
        pagamentos = _collect(fut_map["pagamentos"], "pagamentos")
        estornos   = _collect(fut_map["estornos"],   "estornos")
        cortesias  = _collect(fut_map["cortesias"],  "cortesias")

        result = "\n\n".join([header, vendas, delivery, pagamentos, estornos, cortesias])
        insight = _generate_insight(result)
        return f"{result}\n\n{insight}" if insight else result

    def _run_por_casa(self, header: str, where: str, where_pagto: str) -> str:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        fut_vendas     = pool.submit(_query_vendas_por_casa,     where)
        fut_delivery   = pool.submit(_query_delivery_por_casa,   where)
        fut_pagamentos = pool.submit(_query_pagamentos_por_casa, where_pagto)
        fut_estornos   = pool.submit(_query_estornos_por_casa,   where)
        fut_cortesias  = pool.submit(_query_cortesias_por_casa,  where)

        concurrent.futures.wait(
            [fut_vendas, fut_delivery, fut_pagamentos, fut_estornos, fut_cortesias],
            timeout=580,
        )
        pool.shutdown(wait=False)

        def _safe_dict(fut, section: str, default):
            if fut.done():
                try:
                    return fut.result()
                except Exception as e:
                    logger.warning("[resumo] Falha em %s_por_casa: %s", section, e)
            else:
                logger.warning("[resumo] Timeout em %s_por_casa — descartando resultado.", section)
            return default

        vendas_blocos, totais = _safe_dict(fut_vendas, "vendas", ({}, {}))
        delivery_blocos   = _safe_dict(fut_delivery,   "delivery",   {})
        pagamentos_blocos = _safe_dict(fut_pagamentos, "pagamentos", {})
        estornos_blocos   = _safe_dict(fut_estornos,   "estornos",   {})
        cortesias_blocos  = _safe_dict(fut_cortesias,  "cortesias",  {})

        # Ordena casas pelo faturamento DESC; casas sem vendas (só em outras seções) vão ao final.
        casas_com_vendas = sorted(totais.keys(), key=lambda c: totais.get(c, 0.0), reverse=True)
        todas_casas = (
            set(totais.keys())
            | set(delivery_blocos.keys())
            | set(pagamentos_blocos.keys())
            | set(estornos_blocos.keys())
            | set(cortesias_blocos.keys())
        )
        casas_sem_vendas = sorted(todas_casas - set(totais.keys()))
        casas_finais = casas_com_vendas + casas_sem_vendas

        if not casas_finais:
            return f"{header}\n\n- Sem dados no período."

        blocos_saida = [header]
        for casa in casas_finais:
            partes = [
                f"*{casa}*",
                vendas_blocos.get(casa,     "*Vendas*\n- Sem dados no período."),
                delivery_blocos.get(casa,   "*Delivery*\n- Sem dados no período."),
                pagamentos_blocos.get(casa, "*Formas de pagamento*\n- Sem dados no período."),
                estornos_blocos.get(casa,   "*Estornos*\n- Sem dados no período."),
                cortesias_blocos.get(casa,  "*Cortesias*\n- Sem dados no período."),
            ]
            blocos_saida.append("\n\n".join(partes))

        return "\n\n———\n\n".join(blocos_saida)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)
