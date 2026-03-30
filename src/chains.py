import concurrent.futures
import hashlib
import logging
import re
import threading
import time
import unicodedata
from datetime import datetime

from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

import redis

from src.config import (
    OPENAI_API_KEY, OPENAI_MODEL_NAME, OPENAI_MODEL_TEMPERATURE, OPENAI_BASE_URL,
    OPENAI_FALLBACK_MODEL,
    SQL_AGENT_MAX_ITERATIONS, SQL_AGENT_MAX_EXECUTION_TIME,
    RAG_AGENT_MAX_ITERATIONS, RAG_AGENT_MAX_EXECUTION_TIME,
    CONVERSATION_MAX_HISTORY,
    REDIS_URL, QUERY_CACHE_TTL,
)
from src.memory import get_session_history
from src.prompts import react_prompt, rag_prompt, router_prompt, general_prompt
from src.tools.dremio_tools import DremioSalesQueryTool, DremioDeliveryQueryTool, DremioPaymentQueryTool, DremioEstornosQueryTool, DremioMetasQueryTool, DremioCortesiasQueryTool, current_sender
from src.tools.mysql_tools import MySQLPurchasesQueryTool
from src.tools.rag_tool import RAGDocumentQueryTool
from src.tools.chart_tool import ChartTool
from src.tools.excel_tool import ExcelExportTool

logger = logging.getLogger(__name__)

_MAX_HISTORY = CONVERSATION_MAX_HISTORY
_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)

_BASES_LISTA = (
    "Tenho acesso as seguintes bases de dados:\n\n"
    "📊 *Vendas* — faturamento, ticket medio, fluxo de pessoas, produtos, funcionarios, descontos e Same Store Sales (SSS)\n\n"
    "🛵 *Delivery* — pedidos e faturamento por plataforma (iFood, Rappi, app proprio)\n\n"
    "↩️ *Estornos* — cancelamentos, devolucoes e motivos por produto/funcionario\n\n"
    "🎯 *Metas* — realizado vs orcado, atingimento e delta por casa\n\n"
    "💳 *Formas de pagamento* — receita por metodo (PIX, cartao, dinheiro, etc.)\n\n"
    "🎁 *Cortesias* — itens cortesia por produto, funcionario, tipo e casa\n\n"
    "🛒 *Compras* — pedidos de compra, fornecedores e notas fiscais de entrada\n\n"
    "📄 *Documentos internos* — politicas, procedimentos, organograma e contatos\n\n"
    "📥 Qualquer resultado pode ser exportado em *Excel* — e so pedir!\n\n"
    "Como posso te ajudar hoje?"
)

_FIRST_CONTACT_INTRO = "Sou o TATUDO AQUI, seu assistente interno.\n\n" + _BASES_LISTA

_BASES_RE = re.compile(
    r'\b(quais (dados|bases|informacoes|informações|base de dados)|'
    r'o que (voce|você) (tem|sabe|acessa|consulta)|'
    r'que (dados|bases|informacoes|informações) (tem|voce tem|você tem)|'
    r'o que (consigo|posso) (perguntar|consultar|pedir)|'
    r'quais (consultas|relatorios|relatórios) (posso|consigo)|'
    r'me (mostra|mostre|lista|liste) (as bases|os dados|o que tem))\b',
    re.IGNORECASE,
)


def _msg_hash(message: str) -> str:
    return hashlib.md5(message.lower().strip().encode()).hexdigest()


def _cache_get(session_id: str, message: str) -> str | None:
    key = f"cache:{session_id}:{_msg_hash(message)}"
    try:
        return _redis.get(key)
    except redis.RedisError:
        return None


def _cache_set(session_id: str, message: str, response: str) -> None:
    key = f"cache:{session_id}:{_msg_hash(message)}"
    try:
        _redis.setex(key, QUERY_CACHE_TTL, response)
        logger.info("Cache gravado para %s (TTL=%ds): %.60s", session_id, QUERY_CACHE_TTL, message)
    except redis.RedisError:
        pass


def _metric_inc(key: str) -> None:
    try:
        _redis.incr(f"metrics:{key}")
    except redis.RedisError:
        pass


def _latency_bucket(elapsed: float) -> str:
    if elapsed < 5:
        return "<5s"
    if elapsed < 30:
        return "5-30s"
    if elapsed < 60:
        return "30-60s"
    return ">60s"
_SALARY_RE = re.compile(
    r'\b(salario|salários|salario|remuneracao|remuneração|quanto ganha|quanto recebe|'
    r'quanto ganhou|quanto recebia|folha de pagamento|folha salarial|holerite|contracheque|'
    r'pagamento de funcionario|pagamento de colaborador|pjt|p\.j\.t)\b',
    re.IGNORECASE,
)
_SALARY_BLOCK_MSG = (
    "Nao tenho acesso a informacoes sobre salarios ou remuneracoes — "
    "esses dados sao sensiveis e nao estao disponiveis para consulta."
)

_DATE_WITHOUT_YEAR = re.compile(r'(?<![/\d])(\d{1,2}/\d{1,2})(?![\d/])')
_DATE_YEAR_EXTRA_DIGITS = re.compile(r'\b(\d{1,2}/\d{1,2}/)(\d{5,})\b')
_EXCEL_RE = re.compile(r'\b(excel|planilha|xlsx)\b', re.IGNORECASE)
_GREETING_RE = re.compile(
    r'^\s*(oi+|ola|olá|eae|eai|e ai|e aí|hey|hi|hello|bom dia|boa tarde|boa noite|'
    r'tudo bem|tudo bom|tudo certo|salve|opa|fala|fala ai|boa|ok|okay)\s*[!?.,]*\s*$',
    re.IGNORECASE,
)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()

_model: ChatOpenAI | None = None
_model_lock = threading.Lock()

_fallback_model: ChatOpenAI | None = None
_fallback_model_lock = threading.Lock()

_sql_executor: AgentExecutor | None = None
_sql_executor_lock = threading.Lock()

_rag_executor: AgentExecutor | None = None
_rag_executor_lock = threading.Lock()

_fallback_sql_executor: AgentExecutor | None = None
_fallback_sql_lock = threading.Lock()

_fallback_rag_executor: AgentExecutor | None = None
_fallback_rag_lock = threading.Lock()


def _get_model() -> ChatOpenAI:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ChatOpenAI(
                    model=OPENAI_MODEL_NAME,
                    temperature=OPENAI_MODEL_TEMPERATURE,
                    base_url=OPENAI_BASE_URL,
                    api_key=OPENAI_API_KEY,
                    default_headers={"X-OpenRouter-Cache": "1"},
                )
    return _model


def _get_fallback_model() -> ChatOpenAI | None:
    """Retorna modelo de fallback se FALLBACK_MODEL_NAME estiver configurado."""
    if not OPENAI_FALLBACK_MODEL:
        return None
    global _fallback_model
    if _fallback_model is None:
        with _fallback_model_lock:
            if _fallback_model is None:
                logger.info("Inicializando modelo de fallback: %s", OPENAI_FALLBACK_MODEL)
                _fallback_model = ChatOpenAI(
                    model=OPENAI_FALLBACK_MODEL,
                    temperature=OPENAI_MODEL_TEMPERATURE,
                    base_url=OPENAI_BASE_URL,
                    api_key=OPENAI_API_KEY,
                    default_headers={"X-OpenRouter-Cache": "1"},
                )
    return _fallback_model


def _make_sql_executor(model: ChatOpenAI) -> AgentExecutor:
    tools = [DremioSalesQueryTool(), DremioDeliveryQueryTool(), DremioPaymentQueryTool(), DremioEstornosQueryTool(), DremioMetasQueryTool(), DremioCortesiasQueryTool(), MySQLPurchasesQueryTool(), ChartTool(), ExcelExportTool()]
    agent = create_react_agent(llm=model, tools=tools, prompt=react_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
            "Se nao precisar usar ferramentas, responda com: "
            "Final Answer: [sua resposta]. Nunca responda sem usar esse formato."
        ),
        max_iterations=SQL_AGENT_MAX_ITERATIONS,
        max_execution_time=SQL_AGENT_MAX_EXECUTION_TIME,
    )


def _make_rag_executor(model: ChatOpenAI) -> AgentExecutor:
    tools = [RAGDocumentQueryTool()]
    agent = create_react_agent(llm=model, tools=tools, prompt=rag_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=(
            "Se nao encontrar a informacao, responda com: "
            "Final Answer: Nao encontrei essa informacao nos documentos disponíveis."
        ),
        max_iterations=RAG_AGENT_MAX_ITERATIONS,
        max_execution_time=RAG_AGENT_MAX_EXECUTION_TIME,
    )


def _get_sql_executor() -> AgentExecutor:
    global _sql_executor
    if _sql_executor is None:
        with _sql_executor_lock:
            if _sql_executor is None:
                logger.info("Inicializando agente SQL...")
                _sql_executor = _make_sql_executor(_get_model())
                logger.info("Agente SQL pronto.")
    return _sql_executor


def _get_rag_executor() -> AgentExecutor:
    global _rag_executor
    if _rag_executor is None:
        with _rag_executor_lock:
            if _rag_executor is None:
                logger.info("Inicializando agente RAG...")
                _rag_executor = _make_rag_executor(_get_model())
                logger.info("Agente RAG pronto.")
    return _rag_executor


def _get_fallback_sql_executor() -> AgentExecutor | None:
    fb = _get_fallback_model()
    if not fb:
        return None
    global _fallback_sql_executor
    if _fallback_sql_executor is None:
        with _fallback_sql_lock:
            if _fallback_sql_executor is None:
                logger.info("Inicializando agente SQL de fallback...")
                _fallback_sql_executor = _make_sql_executor(fb)
    return _fallback_sql_executor


def _get_fallback_rag_executor() -> AgentExecutor | None:
    fb = _get_fallback_model()
    if not fb:
        return None
    global _fallback_rag_executor
    if _fallback_rag_executor is None:
        with _fallback_rag_lock:
            if _fallback_rag_executor is None:
                logger.info("Inicializando agente RAG de fallback...")
                _fallback_rag_executor = _make_rag_executor(fb)
    return _fallback_rag_executor


def _complete_dates(message: str) -> str:
    now = datetime.now()
    message = _DATE_YEAR_EXTRA_DIGITS.sub(lambda m: m.group(1) + m.group(2)[:4], message)

    def _fill_year(match: re.Match) -> str:
        day, month = match.group(1).split('/')
        year = now.year
        try:
            candidate = datetime(year, int(month), int(day))
            if (candidate - now).days > 30:
                year -= 1
        except ValueError:
            pass
        return f"{match.group(1)}/{year}"

    return _DATE_WITHOUT_YEAR.sub(_fill_year, message)


def _build_invoke_input(message: str, history, sender_name: str) -> dict:
    is_first_message = len(history.messages) == 0
    is_pure_greeting = bool(_GREETING_RE.match(message))

    history_text = ""
    if history.messages:
        history_text = "Historico recente da conversa (use para entender continuidade):\n"
        for msg in history.messages[-_MAX_HISTORY:]:
            role = "Usuario" if msg.type == "human" else "Assistente"
            history_text += f"{role}: {msg.content}\n"

    if is_first_message and is_pure_greeting and sender_name:
        sender_context = (
            f"Nome do usuario: {sender_name}. "
            f"Responda APENAS com: 'Oi, {sender_name}! {_FIRST_CONTACT_INTRO}'"
        )
    elif is_first_message and is_pure_greeting:
        sender_context = f"Responda APENAS com: 'Oi! {_FIRST_CONTACT_INTRO}'"
    elif sender_name:
        sender_context = f"Nome do usuario no WhatsApp: {sender_name}."
    else:
        sender_context = ""

    return {
        "input": message,
        "current_date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "sender_context": sender_context,
        "history": history_text,
    }


def _trim_history(history) -> None:
    all_msgs = history.messages
    if len(all_msgs) > _MAX_HISTORY * 2:
        keep = all_msgs[-(_MAX_HISTORY * 2):]
        history.clear()
        for msg in keep:
            if msg.type == "human":
                history.add_user_message(msg.content)
            else:
                history.add_ai_message(msg.content)


_ERROR_PREFIXES = (
    "Desculpe, ocorreu um erro",
    "Desculpe, nao consegui processar",
    "Nao encontrei informacoes",
    "Desculpe, ocorreu um erro ao consultar",
    "Nao consegui obter",
)


def _is_error_response(response: str) -> bool:
    return any(response.strip().startswith(p) for p in _ERROR_PREFIXES)


def _save_to_history(message: str, response: str, session_id: str, history=None) -> None:
    import time
    from langchain_core.messages import HumanMessage, AIMessage

    if _is_error_response(response):
        logger.info("Resposta de erro — nao salva no historico de %s.", session_id)
        return
    if history is None:
        history = get_session_history(session_id)
    ts = time.time()
    history.add_message(HumanMessage(content=message, additional_kwargs={"timestamp": ts}))
    history.add_message(AIMessage(content=response, additional_kwargs={"timestamp": ts}))
    _trim_history(history)
    logger.debug("Historico de %s atualizado (%d mensagens).", session_id, len(history.messages))


def _generate_excel_filename(session_id: str) -> str:
    """Gera nome de arquivo descritivo baseado na última pergunta do histórico."""
    try:
        history = get_session_history(session_id)
        last_question = ""
        for msg in reversed(history.messages):
            if msg.type == "human":
                last_question = msg.content
                break
        if not last_question:
            return f"dados_{datetime.now().strftime('%d_%m_%Y')}.xlsx"
        result = _get_model().invoke(
            f"Gere APENAS o nome de um arquivo .xlsx (sem extensao, sem aspas, sem explicacao) "
            f"descrevendo os dados desta consulta: '{last_question}'. "
            f"Use underscores, datas no formato DD_MM_AAAA, sem acentos. "
            f"Exemplos: vendas_restaurantes_16_03_a_22_03_2026 | compras_bebidas_TB_16_03_a_22_03_2026 | "
            f"delivery_ifood_marco_2026. Responda SOMENTE o nome do arquivo."
        )
        name = result.content.strip().replace(" ", "_").replace("/", "_")
        name = re.sub(r'[^\w\-]', '', name)
        if not name:
            return f"dados_{datetime.now().strftime('%d_%m_%Y')}.xlsx"
        return f"{name}.xlsx"
    except Exception as e:
        logger.warning("[excel-fastpath] Falha ao gerar nome do arquivo: %s", e)
        return f"dados_{datetime.now().strftime('%d_%m_%Y')}.xlsx"


def _run_sql_agent(message: str, session_id: str, sender_name: str, history=None) -> str:
    current_sender.set(session_id)
    if history is None:
        history = get_session_history(session_id)
    history_len = len(history.messages)
    logger.info("[sql-agent] session=%s | historico=%d msgs | pergunta: %s", session_id, history_len, message)
    invoke_input = _build_invoke_input(message, history, sender_name)
    try:
        result = _get_sql_executor().invoke(invoke_input)
        output = result.get('output', '')
        if not output or 'Agent stopped' in output or 'iteration limit' in output.lower():
            logger.warning("[sql-agent] Parou por limite de iteracoes. Output: %r", output)
            return 'Desculpe, nao consegui processar sua pergunta a tempo. Tente reformular ou seja mais especifico.'
        logger.info("[sql-agent] Resposta gerada (%.120s%s)", output, '...' if len(output) > 120 else '')
        return output
    except Exception as e:
        logger.error("[sql-agent] Excecao inesperada: %s — tentando fallback", e)
        fb = _get_fallback_sql_executor()
        if fb:
            try:
                logger.info("[sql-agent] Usando modelo de fallback...")
                result = fb.invoke(invoke_input)
                output = result.get('output', '')
                if output and 'Agent stopped' not in output:
                    logger.info("[sql-agent] Fallback respondeu.")
                    return output
            except Exception as e2:
                logger.error("[sql-agent] Fallback tambem falhou: %s", e2)
        return 'Desculpe, ocorreu um erro ao processar sua pergunta.'


def _run_rag_agent(message: str, session_id: str, sender_name: str, history=None) -> str:
    if history is None:
        history = get_session_history(session_id)
    logger.info("[rag-agent] session=%s | pergunta: %s", session_id, message)
    invoke_input = _build_invoke_input(message, history, sender_name)
    try:
        result = _get_rag_executor().invoke(invoke_input)
        output = result.get('output', '')
        if not output or 'Agent stopped' in output:
            logger.warning("[rag-agent] Parou por limite. Output: %r", output)
            return 'Nao encontrei informacoes nos documentos disponíveis.'
        logger.info("[rag-agent] Resposta gerada (%.120s%s)", output, '...' if len(output) > 120 else '')
        return output
    except Exception as e:
        logger.error("[rag-agent] Excecao inesperada: %s — tentando fallback", e)
        fb = _get_fallback_rag_executor()
        if fb:
            try:
                logger.info("[rag-agent] Usando modelo de fallback...")
                result = fb.invoke(invoke_input)
                output = result.get('output', '')
                if output and 'Agent stopped' not in output:
                    logger.info("[rag-agent] Fallback respondeu.")
                    return output
            except Exception as e2:
                logger.error("[rag-agent] Fallback tambem falhou: %s", e2)
        return 'Desculpe, ocorreu um erro ao consultar os documentos.'


def generate_thinking_message(message: str) -> str:
    """Gera uma mensagem de espera contextual e amigavel baseada na pergunta do usuario."""
    try:
        prompt = (
            "Voce e o TATUDO AQUI, assistente interno de uma empresa de bares e restaurantes. "
            "O usuario acabou de fazer a seguinte pergunta:\n"
            f'"{message}"\n\n'
            "Escreva UMA frase curta e natural em portugues avisando que ja vai buscar essa informacao. "
            "REGRAS:\n"
            "- Varie MUITO o estilo a cada vez: as vezes mais animado, as vezes mais direto, as vezes com humor leve, as vezes mais formal\n"
            "- Varie o inicio da frase: nunca comece sempre com 'Claro', 'Ja' ou 'Vou'. Use tambem: 'Deixa eu', 'Um segundo', 'Boa pergunta', 'Ja resolvo', 'To buscando', 'Boa', 'Perfeito', 'Otimo', 'Me da um instante', 'Agora eu vejo', entre outros\n"
            "- Conecte com o tema da pergunta quando possivel (ex: se for sobre vendas, mencione vendas; se for delivery, mencione delivery)\n"
            "- Pode usar expressoes informais e brasileiras naturais\n"
            "- NUNCA use emojis\n"
            "- NUNCA use diminutivos (ex: rapidinho, agorinha, pouquinho, instaninho). Use sempre a forma plena das palavras\n"
            "- NUNCA use mais de uma frase\n"
            "- NUNCA mencione banco de dados, tabela, sistema ou tecnologia"
        )
        model = _get_fallback_model() or _get_model()
        result = model.invoke(prompt)
        return result.content.strip()
    except Exception as e:
        logger.warning("Falha ao gerar mensagem de espera dinamica: %s", e)
        return "Ja vou buscar essa informacao para voce."


def _run_general_response(message: str, session_id: str, sender_name: str, history=None) -> str:
    if history is None:
        history = get_session_history(session_id)
    invoke_input = _build_invoke_input(message, history, sender_name)
    try:
        prompt_text = general_prompt.format(**invoke_input)
        model = _get_fallback_model() or _get_model()
        result = model.invoke(prompt_text)
        return result.content
    except Exception as e:
        logger.error("Erro na resposta geral: %s", e)
        return "Olá! Como posso ajudar?"


_VALID_CATEGORIES = ("sql", "docs", "ambos", "geral")


def _classify_intent(message: str, history_text: str = "") -> str:
    try:
        history_section = f"Historico recente:\n{history_text}\n" if history_text else ""
        result = _get_model().invoke(router_prompt.format(input=message, history=history_section))
        raw = result.content.strip().lower()

        # Grok as vezes retorna "Categoria: sql" ou "sql." ou texto extra — extrai a categoria
        for cat in _VALID_CATEGORIES:
            if cat in raw:
                return cat

        logger.warning("Router retornou categoria invalida '%s', usando 'geral'", raw)
        return "geral"
    except Exception as e:
        logger.error("Erro no router: %s — usando 'geral' como fallback", e)
        return "geral"


def invoke_sql_agent(message: str, session_id: str, sender_name: str = "") -> str:
    message = _complete_dates(message)
    response = _strip_emojis(_run_sql_agent(message, session_id, sender_name))
    _save_to_history(message, response, session_id)
    return response


def invoke_rag_agent(message: str, session_id: str, sender_name: str = "") -> str:
    response = _strip_emojis(_run_rag_agent(message, session_id, sender_name))
    _save_to_history(message, response, session_id)
    return response


def route_and_invoke(message: str, session_id: str, sender_name: str = "", on_thinking=None) -> str:
    message = _complete_dates(message)

    # Fast-path: pergunta sobre quais dados/bases estão disponíveis
    if _BASES_RE.search(message):
        _metric_inc("category:bases_info")
        logger.info("Pergunta sobre bases de %s — resposta direta sem LLM", session_id)
        _save_to_history(message, _BASES_LISTA, session_id)
        return _BASES_LISTA

    # Bloqueio de dados sensíveis: salário / remuneração
    if _SALARY_RE.search(message):
        _metric_inc("blocked:salary")
        logger.info("Pergunta sobre salario bloqueada para %s: %.80s", session_id, message)
        return _SALARY_BLOCK_MSG

    # Fast-path: pedido de Excel sobre dados já consultados
    # Só usa o cache se o DataFrame tiver colunas suficientes (>= 4).
    # DataFrames com poucas colunas são consultas agregadas (ex: só Fantasia + total)
    # e precisam que o agente gere uma query mais detalhada com datas, grupos, etc.
    if _EXCEL_RE.search(message):
        from src.tools.excel_tool import get_last_df, df_to_excel_marker
        last_df = get_last_df(session_id)
        if last_df is not None and len(last_df.columns) >= 4:
            filename = _generate_excel_filename(session_id)
            marker = df_to_excel_marker(last_df, filename)
            response = f"{marker}\nPlanilha gerada com os dados da ultima consulta!"
            _metric_inc("category:excel_fastpath")
            logger.info("[excel-fastpath] Usando ultimo DataFrame da sessao %s (%d linhas, %d colunas) | arquivo=%s", session_id, len(last_df), len(last_df.columns), filename)
            _save_to_history(message, response, session_id)
            return response
        logger.info("[excel-fastpath] DataFrame com %d colunas — passando para o agente gerar query detalhada.", len(last_df.columns) if last_df is not None else 0)

    # Fast-path: saudações simples não precisam do router nem do agente
    if _GREETING_RE.match(message):
        _metric_inc("category:geral")
        history = get_session_history(session_id)
        is_first = len(history.messages) == 0
        if is_first:
            # Primeiro contato: resposta determinística, sem LLM
            nome = f", {sender_name}" if sender_name else ""
            response = f"Oi{nome}! {_FIRST_CONTACT_INTRO}"
            logger.info("Saudacao inicial de %s — resposta direta sem LLM", session_id)
        else:
            logger.info("Saudacao de %s — gerando resposta via LLM", session_id)
            response = _strip_emojis(_run_general_response(message, session_id, sender_name, history=history))
        _save_to_history(message, response, session_id, history=history)
        return response

    cached = _cache_get(session_id, message)
    if cached:
        logger.info("Cache hit para %s: %.80s", session_id, message)
        _metric_inc("cache_hits")
        return cached

    _metric_inc("requests_total")
    history = get_session_history(session_id)
    is_first_message = len(history.messages) == 0
    history_text = ""
    if history.messages:
        lines = []
        for msg in history.messages[-4:]:
            role = "Usuario" if msg.type == "human" else "Assistente"
            lines.append(f"{role}: {msg.content}")
        history_text = "\n".join(lines)
    category = _classify_intent(message, history_text)
    logger.info("Intencao classificada como '%s' para: %.80s", category, message)
    _metric_inc(f"category:{category}")

    if category != "geral" and on_thinking:
        try:
            on_thinking()
        except Exception as e:
            logger.warning("Falha ao enviar mensagem de espera: %s", e)

    t_start = time.time()

    if category == "sql":
        response = _run_sql_agent(message, session_id, sender_name, history=history)
        elapsed = time.time() - t_start
        logger.info("Agente SQL respondeu em %.1fs", elapsed)
        _metric_inc(f"latency:sql:{_latency_bucket(elapsed)}")
        if response.startswith("Desculpe"):
            _metric_inc("errors:sql")
    elif category == "docs":
        response = _run_rag_agent(message, session_id, sender_name, history=history)
        elapsed = time.time() - t_start
        logger.info("Agente RAG respondeu em %.1fs", elapsed)
        _metric_inc(f"latency:rag:{_latency_bucket(elapsed)}")
        if response.startswith("Desculpe") or "Nao encontrei" in response:
            _metric_inc("errors:rag")
    elif category == "ambos":
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            sql_future = pool.submit(_run_sql_agent, message, session_id, sender_name)
            rag_future = pool.submit(_run_rag_agent, message, session_id, sender_name)
            sql_resp = sql_future.result()
            docs_resp = rag_future.result()
        response = f"{sql_resp}\n\n---\n\n{docs_resp}"
        elapsed = time.time() - t_start
        logger.info("Agente AMBOS respondeu em %.1fs (paralelo)", elapsed)
    else:  # geral
        response = _run_general_response(message, session_id, sender_name)

    response = _strip_emojis(response)

    if is_first_message:
        nome = f", {sender_name}" if sender_name else ""
        intro = f"Oi{nome}! Sou o TATUDO AQUI, seu assistente interno.\n\n"
        response = intro + response

    if category != "geral" and not _is_error_response(response):
        _cache_set(session_id, message, response)

    _save_to_history(message, response, session_id, history=history)
    return response
