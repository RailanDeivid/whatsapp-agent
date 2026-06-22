import asyncio
import logging
import threading
import time
from contextvars import ContextVar

from langchain.tools import BaseTool

from src.connectors.dremio import client
from src.config import DREMIO_MAX_CONCURRENT
from src.tools.fantasia_abreviacao import ABREVIACAO_TO_FANTASIA
from src.tools.utils import strip_markdown, format_df

logger = logging.getLogger(__name__)

# Número do remetente atual — setado em chains.py antes de invocar o agente
current_sender: ContextVar[str] = ContextVar("current_sender", default="")

_dremio_semaphore = threading.Semaphore(DREMIO_MAX_CONCURRENT)


_SEMAPHORE_TIMEOUT = 60  # segundos máximos na fila antes de desistir


def _run_dremio_query(label: str, query: str) -> str:
    """Executa query no Dremio e retorna resultado formatado. Compartilhado por todas as ferramentas Dremio."""
    query = strip_markdown(query)
    query_original = query
    query = _replace_casa_names_in_dremio_query(query)
    if query != query_original:
        logger.info("[%s] Nomes de casas substituidos na query Dremio.", label)

    if not _dremio_semaphore.acquire(blocking=False):
        sender = current_sender.get()
        if sender:
            try:
                from src.integrations.evolution_api import send_whatsapp_message
                send_whatsapp_message(sender, "Ha outras consultas em andamento. Sua consulta sera processada em breve, aguarde.")
            except Exception as e:
                logger.warning("[%s] Falha ao notificar usuario sobre fila: %s", label, e)
        logger.info("[%s] Aguardando vaga no semaforo Dremio (max_concurrent=%d, timeout=%ds)...", label, DREMIO_MAX_CONCURRENT, _SEMAPHORE_TIMEOUT)
        acquired = _dremio_semaphore.acquire(blocking=True, timeout=_SEMAPHORE_TIMEOUT)
        if not acquired:
            logger.error("[%s] Timeout aguardando semaforo Dremio apos %ds — consultas travadas ou sobrecarga. max_concurrent=%d", label, _SEMAPHORE_TIMEOUT, DREMIO_MAX_CONCURRENT)
            return "Nao foi possivel processar sua consulta — ha muitas consultas em andamento. Tente novamente em instantes."

    try:
        return _execute_dremio_query(label, query)
    finally:
        _dremio_semaphore.release()


def _execute_dremio_query(label: str, query: str) -> str:
    for attempt in range(1, 3):
        logger.info("[%s] Executando query (tentativa %d/2): %.200s", label, attempt, query)
        t0 = time.time()
        try:
            df = client(query)
            elapsed = time.time() - t0
            if df.empty:
                logger.info("[%s] Query retornou 0 linhas em %.1fs.", label, elapsed)
                return "Nenhum resultado encontrado."
            logger.info("[%s] %d linhas x %d colunas retornadas em %.1fs.", label, len(df), len(df.columns), elapsed)
            session_id = current_sender.get()
            if session_id:
                from src.tools.excel_tool import store_last_df
                store_last_df(session_id, df)
            return format_df(df)
        except TimeoutError:
            elapsed = time.time() - t0
            logger.warning("[%s] Timeout na tentativa %d/2 apos %.1fs (max_wait configurado). Query: %.100s", label, attempt, elapsed, query)
            if attempt == 1:
                sender = current_sender.get()
                if sender:
                    try:
                        from src.integrations.evolution_api import send_whatsapp_message
                        send_whatsapp_message(sender, "A consulta esta demorando mais que o esperado. Tentando novamente...")
                    except Exception as notify_err:
                        logger.warning("[%s] Falha ao notificar usuario sobre retry: %s", label, notify_err)
                logger.info("[%s] Tentando novamente apos timeout...", label)
                continue
            logger.error("[%s] Timeout na segunda tentativa apos %.1fs — abortando. Verifique performance da view no Dremio.", label, elapsed)
            return "Tive um problema tecnico ao rodar a analise. Tente novamente em instantes."
        except RuntimeError as e:
            elapsed = time.time() - t0
            logger.error("[%s] Job Dremio falhou apos %.1fs — %s. Query: %.100s", label, elapsed, e, query)
            return "Dados indisponiveis no periodo."
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("[%s] Erro inesperado apos %.1fs — %s: %s. Query: %.100s", label, elapsed, type(e).__name__, e, query)
            return "Dados indisponiveis no periodo."

def _replace_casa_names_in_dremio_query(query: str) -> str:
    """Substitui nomes abreviados de casas pelo valor exato de unidade no Dremio.
    Suporta tanto correspondências exatas quanto padrões LIKE ('%nome%').
    """
    import re as _re

    def _replace(match: _re.Match) -> str:
        quote = match.group(1)
        raw_value = match.group(2)
        prefix = raw_value[:len(raw_value) - len(raw_value.lstrip('%'))]
        suffix = raw_value[len(raw_value.rstrip('%')):]
        stripped = raw_value.strip('%').strip().upper()
        if stripped in ABREVIACAO_TO_FANTASIA:
            return quote + prefix + ABREVIACAO_TO_FANTASIA[stripped] + suffix + quote
        return quote + raw_value + quote

    query = _re.sub(r"(['])([^']*)\1", _replace, query)
    query = _re.sub(r'(["])([^"]*)\1', _replace, query)
    return query


_CODIGO_CASA_HINT = (
    "IMPORTANTE: o campo `codigo_casa` armazena os códigos abreviados das casas DIRETAMENTE. "
    "Use SEMPRE o código abreviado no SQL, NUNCA expanda para o nome completo. "
    "Exemplos de códigos válidos: "
    + ", ".join(ABREVIACAO_TO_FANTASIA.keys())
    + ". "
)


class DremioTransacoesQueryTool(BaseTool):
    name: str = "consultar_transacoes"
    description: str = (
        "ANALISES SUPORTADAS: faturamento total ou por casa/vertical/marca/categoria, ticket medio, fluxo de pessoas, "
        "ranking de produtos mais vendidos, mix de vendas por categoria (alimentos/bebidas/vinhos), "
        "evolucao temporal (dia a dia, semanal, mensal, anual), participacao percentual por dimensao, "
        "variacao vs periodo anterior (crescimento/queda), vendas por horario ou ocasiao (almoco/jantar), "
        "desempenho por funcionario, SSS (Same Store Sales), CMC% sobre vendas. "
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre vendas, faturamento, receita, "
        "ticket medio, fluxo de pessoas, descontos, produtos vendidos, funcionarios, horario de vendas, "
        "vendas por casa, vendas por categoria (alimentos, bebidas, vinhos), vendas por tipo de produto "
        "(chop, cerveja, drink, suco), vendas por segmento (alcoolicos, nao alcoolicos), vendas por marca. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: vendeu, vendas, faturou, faturamento, receita, "
        "quanto foi, quanto vendeu, ticket, fluxo, pessoas, desconto, produto vendido, "
        "alimentos, bebidas, vinhos, chop, cerveja, drink, suco, alcoolico, nao alcoolico, "
        "marca, marcas, por marca, faturamento por marca, vendas por marca. "
        "NUNCA responda com dados de vendas sem antes chamar esta ferramenta. "
        "NUNCA invente valores — use SOMENTE os dados retornados pela ferramenta. "
        "Executa SQL no Dremio. Tabela: views.\"ANALYTICS\".\"fTransacoes\". "
        "SEMPRE agrupe as queries para trazer resultado limpo e direto. "
        "Colunas disponíveis: "
        "unidade (TEXT, nome da CASA/estabelecimento), "
        "segmento (TEXT, vertical/segmento. Valores: Tipo_A, Tipo_B, Tipo_C), "
        "marca (TEXT, marca/bandeira do estabelecimento — use para filtrar ou agrupar por marca; segue a mesma logica de agrupamento que segmento/vertical/BU), "
        "data_evento (DATE, data da venda), "
        "hora_item (FLOAT, hora do item. ORDER BY CASE WHEN hora_item < 6 THEN hora_item + 24 ELSE hora_item END para sequencia 06:00-05:00), "
        "descricao_produto (TEXT, nome do produto vendido — use ilike(descricao_produto, '%termo%') para busca por nome de produto), "
        "quantidade (FLOAT, quantidade vendida), "
        "valor_produto (DOUBLE, valor unitário), "
        "nome_funcionario (TEXT, nome do funcionário), "
        "valor_liquido (DOUBLE, valor líquido final — SEMPRE use este para totais de faturamento), "
        "desconto_total (FLOAT, desconto total aplicado), "
        "qtd_pessoas (FLOAT, SUM desta coluna = Fluxo de pessoas), "
        "media_ticket (NAO e coluna — calcular: SUM(valor_liquido) / SUM(qtd_pessoas)), "
        "Grande_Grupo (TEXT, categoria ampla. Valores: ALIMENTOS, BEBIDAS, VINHOS, OUTRAS COMPRAS), "
        "Grupo (TEXT, tipo específico: SUCOS, CERVEJAS, CHOPS, DRINKS, COQUETEIS, AGUAS, etc.), "
        "Sub_Grupo (TEXT, segmento: ALCOOLICAS, NAO ALCOOLICAS, PRODUTOS DE EVENTO, VENDAS DE ALIMENTOS, etc.). "
        + _CODIGO_CASA_HINT
        + "SINTAXE DE DATAS no Dremio: DATE_SUB(CURRENT_DATE, 1) para ontem; DATE_TRUNC('month', CURRENT_DATE) para inicio do mes. "
        "NUNCA use CURRENT_DATE - INTERVAL nem CURRENT_DATE - 1. "
        "SEMANA FECHADA: calcule as datas exatas (segunda a domingo) e use BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD'. "
        "AGRUPAMENTO TEMPORAL NO DREMIO — a granularidade do GROUP BY deve corresponder EXATAMENTE ao periodo pedido pelo usuario: "
        "dia a dia → data_evento AS data; "
        "por semana → TO_CHAR(DATE_TRUNC('week', data_evento), 'WW-YYYY') AS semana_ano; "
        "por mes → TO_CHAR(DATE_TRUNC('month', data_evento), 'MM-YYYY') AS mes_ano; "
        "por ano → TO_CHAR(data_evento, 'YYYY') AS ano. "
        "REGRA CRITICA DE GRANULARIDADE: se o usuario pedir 'por ano', 'acumulado por ano', 'anual' — GROUP BY APENAS pelo ano, NUNCA inclua coluna de data diaria. "
        "Se pedir 'por mes', GROUP BY apenas pelo mes. Se pedir 'por dia' ou 'dia a dia', GROUP BY pela data. "
        "Adicionar coluna de data diaria quando o usuario pediu agregado anual/mensal e um ERRO grave — o resultado ficara diario em vez de acumulado. "
        "No GROUP BY: use SEMPRE posicoes ordinais (1, 2, 3...). "
        "Exemplo de query por ano com filtro: SELECT * FROM (SELECT TO_CHAR(data_evento, 'YYYY') AS ano, unidade, SUM(qtd_pessoas) AS fluxo FROM tabela GROUP BY 1, 2) WHERE ano IN ('2023','2024') ORDER BY ano, unidade. "
        "DIA DA SEMANA no Dremio: NUNCA use DAY_OF_WEEK() — nao e suportado. "
        "Use EXTRACT(DOW FROM data_evento) que retorna: 1=Domingo, 2=Segunda-feira, 3=Terca-feira, 4=Quarta-feira, 5=Quinta-feira, 6=Sexta-feira, 7=Sabado. "
        "Exemplo CASE para rotular: CASE EXTRACT(DOW FROM data_evento) WHEN 2 THEN 'Segunda-feira' WHEN 3 THEN 'Terca-feira' WHEN 4 THEN 'Quarta-feira' WHEN 5 THEN 'Quinta-feira' WHEN 6 THEN 'Sexta-feira' WHEN 7 THEN 'Sabado' WHEN 1 THEN 'Domingo' END AS dia_semana. "
        "Para ordenar seg a dom: ORDER BY CASE EXTRACT(DOW FROM data_evento) WHEN 2 THEN 1 WHEN 3 THEN 2 WHEN 4 THEN 3 WHEN 5 THEN 4 WHEN 6 THEN 5 WHEN 7 THEN 6 WHEN 1 THEN 7 END. "
        "NUNCA use DATE_FORMAT (MySQL) no Dremio. "
        "OBRIGATORIO: SQL 100% valido para Dremio. Input: query SQL valida para Dremio."
    )

    def _run(self, query: str) -> str:
        return _run_dremio_query("vendas", query)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)


class DremioDeliveryQueryTool(BaseTool):
    name: str = "consultar_delivery"
    description: str = (
        "ANALISES SUPORTADAS: faturamento delivery por plataforma (iFood/Rappi/app proprio), participacao de cada "
        "canal no total, evolucao temporal do delivery, ticket medio por plataforma, mix de produtos no delivery, "
        "comparativo delivery vs salao, fluxo de pedidos por horario ou ocasiao, delivery por marca. "
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre DELIVERY — "
        "pedidos delivery, faturamento delivery, vendas delivery, plataformas de entrega (iFood, Rappi, app proprio), "
        "delivery por marca. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: delivery, entrega, ifood, rappi, app proprio, pedido delivery, "
        "faturamento delivery, vendas delivery, canal de entrega, plataforma, "
        "marca, marcas, por marca, delivery por marca. "
        "NUNCA responda com dados de delivery sem antes chamar esta ferramenta. "
        "NUNCA invente valores — use SOMENTE os dados retornados pela ferramenta. "
        "Executa SQL no Dremio. Tabela: views.\"ANALYTICS\".\"fDelivery\". "
        "SEMPRE agrupe as queries para trazer resultado limpo e direto. "
        "Colunas disponíveis: "
        "unidade (TEXT, nome da CASA/estabelecimento), "
        "segmento (TEXT, vertical/segmento. Valores: Tipo_A, Tipo_B, Tipo_C), "
        "marca (TEXT, marca/bandeira do estabelecimento — use para filtrar ou agrupar por marca; segue a mesma logica de agrupamento que segmento/vertical/BU), "
        "data_evento (DATE, data do pedido delivery), "
        "hora_item (FLOAT, hora do item. ORDER BY CASE WHEN hora_item < 6 THEN hora_item + 24 ELSE hora_item END), "
        "codigo_produto (TEXT, código do produto), "
        "descricao_produto (TEXT, nome do produto vendido), "
        "quantidade (FLOAT, quantidade de itens), "
        "valor_produto (DOUBLE, valor unitário), "
        "valor_bruto (DOUBLE, valor antes de descontos), "
        "desconto_produto (FLOAT, desconto no produto), "
        "desconto_total (FLOAT, desconto total no pedido), "
        "nome_funcionario (TEXT, canal/plataforma: IFOOD, RAPPI, APP PROPRIO, TERMINAL — use para agrupar por plataforma), "
        "valor_conta (DOUBLE, valor total da conta/pedido), "
        "valor_liquido (DOUBLE, valor líquido final — SEMPRE use para totais de faturamento), "
        "qtd_pessoas (FLOAT, SUM desta coluna = Fluxo de pessoas), "
        "media_ticket (NAO e coluna — calcular: SUM(valor_liquido) / SUM(qtd_pessoas)), "
        "Grande_Grupo (TEXT, categoria ampla: ALIMENTOS, BEBIDAS, VINHOS, OUTRAS COMPRAS), "
        "Grupo (TEXT, tipo específico: SUCOS, CERVEJAS, CHOPS, DRINKS, COQUETEIS, AGUAS, etc.), "
        "Sub_Grupo (TEXT, segmento: ALCOOLICAS, NAO ALCOOLICAS, PRODUTOS DE EVENTO, VENDAS DE ALIMENTOS, etc.). "
        + _CODIGO_CASA_HINT
        + "SINTAXE DE DATAS no Dremio: DATE_SUB(CURRENT_DATE, 1) para ontem, "
        "DATE_TRUNC('month', CURRENT_DATE) para inicio do mes. "
        "NUNCA use CURRENT_DATE - INTERVAL nem CURRENT_DATE - 1. "
        "SEMANA FECHADA: calcule datas exatas (segunda a domingo) e use BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD'. "
        "No GROUP BY: use SEMPRE posicoes ordinais (1, 2, 3...). "
        "Exemplo de query com filtro de data: SELECT data_evento AS data, unidade, SUM(valor_liquido) AS total FROM views.\"ANALYTICS\".\"fDelivery\" WHERE data_evento BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD' GROUP BY 1, 2 ORDER BY 1. "
        "OBRIGATORIO: SQL 100% valido para Dremio. "
        "Input: query SQL valida para Dremio."
    )

    def _run(self, query: str) -> str:
        return _run_dremio_query("delivery", query)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)


class DremioEstornosQueryTool(BaseTool):
    name: str = "consultar_estornos"
    description: str = (
        "ANALISES SUPORTADAS: total de estornos por casa/periodo/marca, ranking de motivos de estorno, "
        "produtos mais estornados, estornos por funcionario, percentual de perda (perda=1), "
        "evolucao de estornos ao longo do tempo, estornos por categoria de produto, estornos por marca. "
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre ESTORNOS, "
        "cancelamentos ou devoluções de produtos. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: estorno, estornos, cancelamento, cancelamentos, "
        "devolucao, devolveu, item cancelado, produto cancelado, perda, motivo de estorno, "
        "quanto foi estornado, total de estornos, "
        "marca, marcas, por marca, estornos por marca. "
        "NUNCA responda com dados de estornos sem antes chamar esta ferramenta. "
        "NUNCA invente valores — use SOMENTE os dados retornados pela ferramenta. "
        "Executa SQL no Dremio. Tabela: views.\"ANALYTICS\".\"fEstornos\". "
        "SEMPRE agrupe as queries para trazer resultado limpo e direto. "
        "Colunas disponíveis: "
        "unidade (TEXT, nome da CASA/estabelecimento), "
        "segmento (TEXT, vertical/segmento. Valores: Tipo_A, Tipo_B, Tipo_C), "
        "marca (TEXT, marca/bandeira do estabelecimento — use para filtrar ou agrupar por marca; segue a mesma logica de agrupamento que segmento/vertical/BU), "
        "data_evento (DATE, data do estorno), "
        "codigo_produto (INT, código do produto estornado), "
        "descricao_produto (TEXT, nome do produto estornado), "
        "quantidade (FLOAT, quantidade estornada), "
        "valor_produto (DOUBLE, valor do estorno — use SUM(valor_produto) para totalizar), "
        "descricao_motivo_estorno (TEXT, motivo — use GROUP BY para agrupar por motivo), "
        "perda (INT, 1=sim 0=nao), "
        "tipo_estorno (TEXT, ex: COM FATURAMENTO), "
        "nome_cliente (TEXT, identificação do cliente), "
        "nome_funcionario (TEXT, funcionário que realizou o estorno), "
        "nome_usuario_funcionario (TEXT, login do funcionário), "
        "Grande_Grupo (TEXT, categoria ampla: ALIMENTOS, BEBIDAS, VINHOS, OUTRAS COMPRAS), "
        "Grupo (TEXT, tipo específico: SUCOS, CERVEJAS, CHOPS, DRINKS, etc.), "
        "Sub_Grupo (TEXT, segmento: ALCOOLICAS, NAO ALCOOLICAS, PRODUTOS DE EVENTO, etc.). "
        + _CODIGO_CASA_HINT
        + "SINTAXE DE DATAS no Dremio: DATE_SUB(CURRENT_DATE, 1) para ontem, "
        "DATE_TRUNC('month', CURRENT_DATE) para inicio do mes. "
        "NUNCA use CURRENT_DATE - INTERVAL nem CURRENT_DATE - 1. "
        "SEMANA FECHADA: calcule datas exatas (segunda a domingo) e use BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD'. "
        "No GROUP BY: use SEMPRE posicoes ordinais (1, 2, 3...). "
        "Exemplo de query com filtro de data: SELECT data_evento AS data, unidade, SUM(valor_produto) AS total FROM views.\"ANALYTICS\".\"fEstornos\" WHERE data_evento BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD' GROUP BY 1, 2 ORDER BY 1. "
        "OBRIGATORIO: SQL 100% valido para Dremio. "
        "Input: query SQL valida para Dremio."
    )

    def _run(self, query: str) -> str:
        return _run_dremio_query("estornos", query)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)


class DremioMetasQueryTool(BaseTool):
    name: str = "consultar_metas"
    description: str = (
        "ANALISES SUPORTADAS: atingimento de meta por casa/segmento/periodo, delta realizado vs orcado (R$ e %), "
        "ranking de casas acima/abaixo da meta, fluxo vs meta de fluxo, evolucao do atingimento ao longo do tempo, "
        "projecao de fechamento de mes (se o usuario pedir). Para qualquer analise que combine realizado vs meta, "
        "use CTE juntando fTransacoes e dMetas em uma unica query. "
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre METAS, orcamento, "
        "budget, atingimento, delta de meta, real vs meta, realizado vs orcado, fluxo vs meta. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: meta, metas, orcamento, budget, atingimento, "
        "atingiu a meta, delta, vs meta, real vs meta, rel vs meta, acima da meta, abaixo da meta, "
        "receita meta, fluxo meta, quanto faltou, quanto sobrou, percentual de atingimento. "
        "NUNCA responda com dados de metas sem antes chamar esta ferramenta. "
        "NUNCA invente valores — use SOMENTE os dados retornados pela ferramenta. "
        "Executa SQL no Dremio. Tabela principal: views.\"ANALYTICS\".\"dMetas\". "
        "ATENCAO: colunas com espaco DEVEM estar entre aspas duplas no SQL. "
        "Colunas disponíveis: "
        "DATA (DATE, data da meta), "
        "\"RECEITA META\" (FLOAT, meta diaria de faturamento — use SUM(\"RECEITA META\") para totalizar), "
        "\"META FLUXO\" (FLOAT, meta diaria de fluxo de pessoas — use SUM(\"META FLUXO\") para totalizar), "
        "unidade (TEXT, nome COMPLETO do estabelecimento — o JOIN com fTransacoes funciona direto), "
        "segmento (TEXT, vertical/segmento. Valores: Tipo_A, Tipo_B, Tipo_C), "
        "marca (TEXT, marca/bandeira do estabelecimento — use para filtrar ou agrupar por marca; segue a mesma logica de agrupamento que segmento/vertical/BU). "
        "PARA 'VENDAS VS META', 'FATURAMENTO VS META', 'REL VS META', 'REAL VS META': "
        "use CTE juntando fTransacoes e dMetas em uma unica query — NUNCA chame consultar_transacoes separadamente. "
        "Padrao CTE obrigatorio: "
        "WITH vendas AS (SELECT unidade, SUM(valor_liquido) AS realizado FROM views.\"ANALYTICS\".\"fTransacoes\" WHERE data_evento BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD' GROUP BY unidade), "
        "metas AS (SELECT unidade, SUM(\"RECEITA META\") AS meta FROM views.\"ANALYTICS\".\"dMetas\" WHERE DATA BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD' GROUP BY unidade) "
        "SELECT v.unidade, v.realizado, m.meta, v.realizado - m.meta AS delta_rs, ROUND((v.realizado - m.meta) / m.meta * 100, 2) AS delta_pct, ROUND(v.realizado / m.meta * 100, 2) AS atingimento_pct FROM vendas v JOIN metas m ON v.unidade = m.unidade ORDER BY atingimento_pct DESC. "
        "Para FLUXO VS META: substitua valor_liquido por SUM(qtd_pessoas) e \"RECEITA META\" por \"META FLUXO\". "
        "Para 'abaixo da meta': adicione WHERE v.realizado < m.meta. Para 'acima da meta': WHERE v.realizado > m.meta. "
        "Para filtrar por segmento/BU: adicione WHERE m.segmento = 'Tipo_A' (ou 'Tipo_B' ou 'Tipo_C') nas duas CTEs. "
        "SINTAXE DE DATAS no Dremio: DATE_SUB(CURRENT_DATE, 1) para ontem, "
        "DATE_TRUNC('month', CURRENT_DATE) para inicio do mes. "
        "NUNCA use CURRENT_DATE - INTERVAL nem CURRENT_DATE - 1. "
        "SEMANA FECHADA: calcule datas exatas (segunda a domingo) e use BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD'. "
        "No GROUP BY: use SEMPRE posicoes ordinais (1, 2, 3...). "
        "Exemplo de query com filtro de data: SELECT DATA AS data, unidade, SUM(\"RECEITA META\") AS meta FROM views.\"ANALYTICS\".\"dMetas\" WHERE DATA BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD' GROUP BY 1, 2 ORDER BY 1. "
        "OBRIGATORIO: SQL 100% valido para Dremio. "
        "Input: query SQL valida para Dremio."
    )

    def _run(self, query: str) -> str:
        return _run_dremio_query("metas", query)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)


class DremioPaymentQueryTool(BaseTool):
    name: str = "consultar_formas_pagamento"
    description: str = (
        "ANALISES SUPORTADAS: mix de formas de pagamento (participacao % de cada meio), evolucao do uso de "
        "pix/cartao/dinheiro ao longo do tempo, comparativo de meios de pagamento entre casas/marcas, "
        "concentracao por forma de pagamento, tendencia de adocao de pagamento digital, pagamentos por marca. "
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre FORMAS DE PAGAMENTO, "
        "mix de pagamentos, participacao por forma de pagamento, faturamento por pagamento. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: forma de pagamento, formas de pagamento, "
        "pagamento, dinheiro, cartao, credito, debito, pix, voucher, mix de pagamento, "
        "quanto foi pago em cartao, quanto em pix, quanto em dinheiro, participacao de pagamento, "
        "marca, marcas, por marca, pagamento por marca. "
        "NUNCA responda com dados de formas de pagamento sem antes chamar esta ferramenta. "
        "NUNCA invente valores — use SOMENTE os dados retornados pela ferramenta. "
        "Executa SQL no Dremio. Tabela: views.\"ANALYTICS\".\"fFormasPagamento\". "
        "SEMPRE agrupe as queries para trazer resultado limpo e direto. "
        "Colunas disponíveis: "
        "cnpj_casa (TEXT, CNPJ do estabelecimento), "
        "unidade (TEXT, nome da CASA/estabelecimento), "
        "segmento (TEXT, vertical/segmento. Valores: Tipo_A, Tipo_B, Tipo_C), "
        "marca (TEXT, marca/bandeira do estabelecimento — use para filtrar ou agrupar por marca; segue a mesma logica de agrupamento que segmento/vertical/BU), "
        "data (DATE, data do registro), "
        "descricao_corretas (TEXT, nome da forma: VISA_CREDITO, DINHEIRO, PIX, etc.), "
        "pessoas (FLOAT, numero de pessoas), "
        "vl_recebido (DOUBLE, valor bruto recebido — use SUM(vl_recebido) para totais). "
        + _CODIGO_CASA_HINT
        + "SINTAXE DE DATAS no Dremio: DATE_SUB(CURRENT_DATE, 1) para ontem, "
        "DATE_TRUNC('month', CURRENT_DATE) para inicio do mes. "
        "NUNCA use CURRENT_DATE - INTERVAL nem CURRENT_DATE - 1. "
        "SEMANA FECHADA: calcule datas exatas (segunda a domingo) e use BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD'. "
        "OBRIGATORIO: SQL 100% valido para Dremio. "
        "Input: query SQL valida para Dremio."
    )

    def _run(self, query: str) -> str:
        return _run_dremio_query("pagamentos", query)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)


class DremioCortesiasQueryTool(BaseTool):
    name: str = "consultar_cortesias"
    description: str = (
        "ANALISES SUPORTADAS: total de cortesias por casa/periodo/marca, ranking de funcionarios que mais concederam "
        "cortesias, produtos mais cortesiados, participacao de cortesias por tipo (desconto conta vs desconto itens), "
        "evolucao temporal de cortesias, cortesias por categoria de produto, percentual de cortesia sobre vendas, cortesias por marca. "
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre CORTESIAS — "
        "itens cortesia, produtos cortesia, cortesia por funcionario, cortesia por produto, "
        "valor de cortesias, quantidade de cortesias, cortesia por casa, por segmento ou por marca. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: cortesia, cortesias, item cortesia, "
        "produto cortesia, cortesia de funcionario, cortesia por produto, total de cortesias, "
        "quanto foi em cortesia, cortesias por periodo, cortesia por tipo, "
        "marca, marcas, por marca, cortesia por marca. "
        "NUNCA responda com dados de cortesias sem antes chamar esta ferramenta. "
        "NUNCA invente valores — use SOMENTE os dados retornados pela ferramenta. "
        "Executa SQL no Dremio. Tabela: views.\"ANALYTICS\".\"fCortesias\". "
        "SEMPRE agrupe as queries para trazer resultado limpo e direto. "
        "Colunas disponíveis: "
        "unidade (TEXT, nome da CASA/estabelecimento), "
        "segmento (TEXT, vertical/segmento. Valores: Tipo_A, Tipo_B, Tipo_C), "
        "marca (TEXT, marca/bandeira do estabelecimento — use para filtrar ou agrupar por marca; segue a mesma logica de agrupamento que segmento/vertical/BU), "
        "codigo_casa (TEXT, codigo abreviado da casa — use para filtros rapidos), "
        "data_evento (DATE, data da venda), "
        "hora_item (INT, hora do lancamento), "
        "terminal (TEXT, terminal/origem: poc, app, ifood, etc.), "
        "nome_funcionario (TEXT, funcionario que registrou a cortesia — use GROUP BY para ranking), "
        "nome_cliente (TEXT, nome do cliente que recebeu a cortesia), "
        "codigo_produto (INT, codigo do produto), "
        "descricao_produto (TEXT, nome do produto — use ilike(descricao_produto, '%termo%') para busca), "
        "descricao_produto_grupo (TEXT, grupo do produto ex: CERVEJAS E CHOPPS), "
        "quantidade (FLOAT, quantidade de itens cortesia), "
        "valor_produto (FLOAT, valor unitario do produto), "
        "valor_bruto (FLOAT, valor de venda do item), "
        "desconto_itens (FLOAT, desconto aplicado no item), "
        "desconto_conta (FLOAT, desconto aplicado na conta), "
        "valor_conta (FLOAT, valor total da conta), "
        "total_cortesias (FLOAT, valor total da cortesia — use SEMPRE ROUND(SUM(total_cortesias), 0) para totalizar cortesias), "
        "tipoDesconto (TEXT, tipo de desconto aplicado — 'Desconto Conta' ou 'Desconto Itens'. Use apenas se o usuario perguntar especificamente por tipo de desconto; retorna na estrutura: * Desconto Itens: R$ xxxx \\n * Desconto Conta: R$ xxxx \\n *Total: R$ xxxx (soma dos dois tipos)), "
        "descricao_cortesias (TEXT, tipo/motivo da cortesia ex: DESCONTO HOLDING — use GROUP BY para agrupar por tipo), "
        "observacao_cortesias (TEXT, observacao livre registrada na cortesia), "
        "Grande_Grupo (TEXT, categoria ampla: ALIMENTOS, BEBIDAS, VINHOS, OUTRAS COMPRAS), "
        "Sub_Grupo (TEXT, subcategoria: CERVEJAS, CHOPS, DRINKS, etc.). "
        + _CODIGO_CASA_HINT
        + "SINTAXE DE DATAS no Dremio: DATE_SUB(CURRENT_DATE, 1) para ontem, "
        "DATE_TRUNC('month', CURRENT_DATE) para inicio do mes. "
        "NUNCA use CURRENT_DATE - INTERVAL nem CURRENT_DATE - 1. "
        "SEMANA FECHADA: calcule datas exatas (segunda a domingo) e use BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD'. "
        "No GROUP BY: use SEMPRE posicoes ordinais (1, 2, 3...). "
        "Exemplo de query com filtro de data: SELECT data_evento AS data, unidade, ROUND(SUM(total_cortesias), 0) AS total FROM views.\"ANALYTICS\".\"fCortesias\" WHERE data_evento BETWEEN 'AAAA-MM-DD' AND 'AAAA-MM-DD' GROUP BY 1, 2 ORDER BY 1. "
        "OBRIGATORIO: SQL 100% valido para Dremio. "
        "Input: query SQL valida para Dremio."
    )

    def _run(self, query: str) -> str:
        return _run_dremio_query("cortesias", query)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)
