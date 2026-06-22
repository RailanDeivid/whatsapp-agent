import asyncio
import logging
import re
import time

from langchain.tools import BaseTool

from src.connectors.mysql import client
from src.tools.fantasia_abreviacao import ABREVIACAO_TO_FANTASIA
from src.tools.utils import strip_markdown

logger = logging.getLogger(__name__)

# CNPJs de PJTs (salário disfarçado) — NUNCA devem aparecer em consultas de compras
_BLOCKED_CNPJS = (
    "'45880307000162'","'48394438000128'","'44760651000155'","'22951189000130'",
    "'34006246000192'","'44533475000119'","'29219157000194'","'44131120000101'",
    "'47959350000143'","'47947110000129'","'27726278000105'","'49863345000168'",
    "'37895137000161'","'30170858000165'","'46293836000122'","'51882614000119'",
    "'43341925000109'","'36163183000103'","'44909381000100'","'46786766000144'",
    "'11513881000160'","'43136020000105'","'44945149000119'","'01360688633'",
)
_PURCHASE_EXCLUSION = (
    "cpf_cnpj_emitente NOT IN (" + ",".join(_BLOCKED_CNPJS) + ")"
    " AND Fantasia NOT IN ('HOLDING')"
    " AND `Descrição Item` NOT IN ('CONSULTORIA')"
)

_WHERE_RE    = re.compile(r'\bWHERE\b', re.IGNORECASE)
_CLAUSE_RE   = re.compile(r'\b(GROUP BY|ORDER BY|LIMIT|HAVING)\b', re.IGNORECASE)


def _inject_exclusion_filter(query: str) -> str:
    """Garante que toda query de compras exclui CNPJs/fantasias/itens sensíveis."""
    query = query.strip().rstrip(';')
    if _WHERE_RE.search(query):
        query = _WHERE_RE.sub(f'WHERE {_PURCHASE_EXCLUSION} AND', query, count=1)
    else:
        m = _CLAUSE_RE.search(query)
        if m:
            query = query[:m.start()] + f'WHERE {_PURCHASE_EXCLUSION} ' + query[m.start():]
        else:
            query = query + f' WHERE {_PURCHASE_EXCLUSION}'
    return query


# Hint compacto incluído na description para o agente LLM já gerar o SQL correto
_ABREV_HINT = (
    "Abreviações válidas para `Fantasia` (use SEMPRE o nome fantasia no SQL, nunca a abreviação): "
    + ", ".join(f"{abr}={fan}" for abr, fan in ABREVIACAO_TO_FANTASIA.items())
    + ". "
)


def _replace_abbreviations_in_query(query: str) -> str:
    """Substitui abreviações por nomes fantasia dentro de literais SQL (segurança extra).
    Suporta tanto correspondências exatas quanto padrões LIKE.
    """
    def _replace(match: re.Match) -> str:
        quote = match.group(1)
        raw_value = match.group(2)
        # Strip wildcards LIKE (% e _) para busca no depara
        prefix = raw_value[:len(raw_value) - len(raw_value.lstrip('%'))]
        suffix = raw_value[len(raw_value.rstrip('%')):]
        stripped = raw_value.strip('%').strip().upper()
        if stripped in ABREVIACAO_TO_FANTASIA:
            return quote + prefix + ABREVIACAO_TO_FANTASIA[stripped] + suffix + quote
        return quote + raw_value + quote

    query = re.sub(r"(['])([^']*)\1", _replace, query)
    query = re.sub(r'(["])([^"]*)\1', _replace, query)
    return query


class MySQLPurchasesQueryTool(BaseTool):
    name: str = "consultar_compras"
    description: str = (
        "ANALISES SUPORTADAS: total comprado por casa/periodo/categoria, ranking de fornecedores por volume, "
        "mix de compras por grande grupo (alimentos/bebidas/vinhos), evolucao temporal de compras, "
        "preco medio simples e ponderado por produto, CMC (Custo da Mercadoria Comprada) — ATENCAO: "
        "nao temos CMV (exige estoque), apenas CMC. CMC% sobre transacoes requer combinar com consultar_transacoes. "
        "FONTE: banco MySQL (DW_DADOS) — use SINTAXE MySQL (DATE_FORMAT, backticks em colunas com espaco). "
        "NUNCA use TO_CHAR, DATE_TRUNC ou funcoes Dremio nesta ferramenta. "
        "QUANDO USAR: OBRIGATORIO chamar esta ferramenta para QUALQUER pergunta sobre COMPRAS, "
        "pedidos de compra, fornecedores ou notas fiscais de entrada. "
        "PALAVRAS-CHAVE que ativam esta ferramenta: compra, compras, comprou, fornecedor, fornecedores, "
        "nota fiscal, NF, pedido de compra, quanto foi comprado, total comprado, custo, "
        "insumo, ingrediente, produto comprado, valor de compra, compra de alimentos, compra de bebidas. "
        "NUNCA responda com dados de compras sem antes chamar esta ferramenta. "
        "NUNCA invente valores — use SOMENTE os dados retornados pela ferramenta. "
        "Executa SQL no MySQL. Banco: DW_DADOS. Tabela: `tb_compras`. "
        "IMPORTANTE: TODOS os nomes de colunas com espacos ou acentos precisam de backticks. "
        "SEMPRE agrupe as queries para trazer resultado limpo e direto. "
        "Colunas disponíveis: "
        "`Fantasia` (TEXT, nome fantasia da casa — SEMPRE pesquise em MAIUSCULO), "
        "`D. Lançamento` (DATE, data oficial da compra/nota), "
        "`N. Nota` (BIGINT, numero da nota fiscal), "
        "`Razão Emitente` (TEXT, razao social do fornecedor), "
        "`Item` (TEXT, codigo do item/produto), "
        "`Descrição Item` (TEXT, nome do produto comprado), "
        "`Grande Grupo` (TEXT, grupo principal: ALIMENTOS, BEBIDAS, VINHOS — use para agrupar por categoria ampla), "
        "`Grupo` (TEXT, subgrupo do produto — use para detalhar por tipo de produto), "
        "`Q. Estoque` (DECIMAL, quantidade em unidade de estoque), "
        "`V. Unitário Convertido` (DECIMAL, valor unitario convertido), "
        "`V. Total` (DECIMAL, valor total da compra — use SUM(`V. Total`) para totalizar). "
        + _ABREV_HINT
        + "BUSCA POR PRODUTO: NUNCA filtre `Descrição Item` com = usando o nome exato do usuario. "
        "Use SEMPRE `Descrição Item` LIKE '%termo%' para busca por nome de produto. "
        + "AGRUPAMENTO TEMPORAL NO MYSQL: "
        "dia a dia → CAST(`D. Lançamento` AS DATE) AS data; "
        "por mes → DATE_FORMAT(`D. Lançamento`, '%m-%Y') AS mes_ano; "
        "por ano → DATE_FORMAT(`D. Lançamento`, '%Y') AS ano. "
        "NUNCA use TO_CHAR ou DATE_TRUNC no MySQL. "
        "Input: query SQL valida para MySQL."
    )

    def _run(self, query: str) -> str:
        query = strip_markdown(query)
        query_original = query
        query = _replace_abbreviations_in_query(query)
        if query != query_original:
            logger.info("[compras] Abreviacoes substituidas na query.")
        query = _inject_exclusion_filter(query)
        logger.info("[compras] Executando query MySQL: %s", query)
        t0 = time.time()
        try:
            df = client(query)
            if df.empty:
                logger.info("[compras] Query retornou 0 linhas em %.1fs.", time.time() - t0)
                return "Nenhum resultado encontrado."
            logger.info("[compras] %d linhas retornadas em %.1fs.", len(df), time.time() - t0)
            from src.tools.dremio_tools import current_sender
            from src.tools.excel_tool import store_last_df
            session_id = current_sender.get()
            if session_id:
                store_last_df(session_id, df)
            return df.to_string(index=False)
        except Exception as e:
            logger.error("[compras] ERRO apos %.1fs — %s: %s", time.time() - t0, type(e).__name__, e)
            return f"Erro ao consultar MySQL (compras): {str(e)}"

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run, query)
