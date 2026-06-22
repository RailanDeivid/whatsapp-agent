import logging
import random
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.message_buffer import buffer_message
from src.integrations.evolution_api import get_media_base64, send_whatsapp_message
from src.integrations.transcribe import transcribe_audio
from src.chains import extract_text_from_image
from src.access_control import init_db, is_authorized, is_admin, authorize, revoke, unblock, delete_user, update_phone, list_users, get_user_nome
from src.memory import clear_session, clear_all_sessions, get_session_messages
from src.config import REDIS_URL, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW, EVOLUTION_AUTHENTICATION_API_KEY, SUPPORT_CONTACT


class _EvolutionKey(BaseModel):
    id: Optional[str] = None
    fromMe: Optional[bool] = False
    remoteJid: Optional[str] = None


class _EvolutionMessage(BaseModel):
    conversation: Optional[str] = None
    extendedTextMessage: Optional[dict] = None
    audioMessage: Optional[dict] = None
    imageMessage: Optional[dict] = None


class _EvolutionData(BaseModel):
    key: Optional[_EvolutionKey] = None
    pushName: Optional[str] = ""
    message: Optional[_EvolutionMessage] = None


class EvolutionWebhookPayload(BaseModel):
    event: Optional[str] = ""
    data: Optional[_EvolutionData] = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Banco de controle de acesso inicializado.")
    yield

app = FastAPI(lifespan=lifespan)
_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)


async def _is_rate_limited(phone: str) -> bool:
    key = f"rl:{phone}"
    # Pipeline garante atomicidade: INCR + EXPIRE na mesma transação Redis
    async with _redis.pipeline(transaction=True) as pipe:
        await pipe.incr(key)
        await pipe.expire(key, RATE_LIMIT_WINDOW)
        count, _ = await pipe.execute()
    return count > RATE_LIMIT_MAX



@app.get("/health")
async def health():
    import asyncio as _asyncio
    checks: dict[str, str] = {}

    # Redis
    try:
        await _redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.error("[health] Redis indisponivel: %s", e)
        checks["redis"] = f"error: {type(e).__name__}"

    # MySQL — testa conexão via pool (bypassa cache para não poluir Redis com SELECT 1)
    try:
        loop = _asyncio.get_running_loop()
        def _mysql_ping():
            from src.connectors.mysql import _get_pool
            db = _get_pool().get_connection()
            cursor = db.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            db.close()
        await loop.run_in_executor(None, _mysql_ping)
        checks["mysql"] = "ok"
    except Exception as e:
        logger.error("[health] MySQL indisponivel: %s", e)
        checks["mysql"] = f"error: {type(e).__name__}"

    # Vectorstore RAG
    try:
        from src.tools.rag_tool import _vectorstore
        checks["vectorstore"] = "ok" if _vectorstore is not None else "not_loaded"
    except Exception as e:
        logger.warning("[health] Vectorstore indisponivel: %s", e)
        checks["vectorstore"] = f"error: {type(e).__name__}"

    overall = "ok" if all(v == "ok" or v == "not_loaded" for v in checks.values()) else "degraded"
    if overall == "degraded":
        logger.warning("[health] Status degradado: %s", checks)
    return {"status": overall, "checks": checks}


@app.get("/metrics")
async def metrics():
    # scan é mais seguro que keys() — não bloqueia Redis em produção com muitas chaves
    all_keys: list[str] = []
    cursor = 0
    while True:
        cursor, batch = await _redis.scan(cursor, match="metrics:*", count=100)
        all_keys.extend(batch)
        if cursor == 0:
            break
    raw: dict[str, int] = {}
    if all_keys:
        values = await _redis.mget(*all_keys)
        raw = {k.removeprefix("metrics:"): int(v) for k, v in zip(all_keys, values) if v}

    requests_total = raw.get("requests_total", 0)
    cache_hits = raw.get("cache_hits", 0)
    cache_hit_rate = round(cache_hits / requests_total * 100, 1) if requests_total else 0.0

    errors_total = sum(v for k, v in raw.items() if k.startswith("errors:"))
    error_rate = round(errors_total / requests_total * 100, 1) if requests_total else 0.0

    latency: dict[str, dict] = {}
    for agent in ("sql", "rag"):
        buckets = {b: raw.get(f"latency:{agent}:{b}", 0) for b in ("<5s", "5-30s", "30-60s", ">60s")}
        total_calls = sum(buckets.values())
        latency[agent] = {**buckets, "total_calls": total_calls}

    categories = {k.removeprefix("category:"): v for k, v in raw.items() if k.startswith("category:")}
    errors = {k.removeprefix("errors:"): v for k, v in raw.items() if k.startswith("errors:")}

    return {
        "resumo": {
            "requests_total": requests_total,
            "cache_hits": cache_hits,
            "cache_hit_rate_pct": cache_hit_rate,
            "errors_total": errors_total,
            "error_rate_pct": error_rate,
        },
        "categorias": categories,
        "latencia": latency,
        "erros": errors,
    }


def _check_admin_key(x_api_key: str | None) -> None:
    """Valida chave de API para endpoints administrativos."""
    if not x_api_key or x_api_key != EVOLUTION_AUTHENTICATION_API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API invalida.")


@app.post("/limpar_cache")
async def limpar_cache(x_api_key: Optional[str] = Header(default=None)):
    """Remove todas as respostas cacheadas do Redis."""
    _check_admin_key(x_api_key)
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = await _redis.scan(cursor, match="cache:*", count=100)
        if keys:
            deleted += await _redis.delete(*keys)
        if cursor == 0:
            break
    logger.info("Cache limpo via endpoint: %d chaves removidas.", deleted)
    return {"chaves_removidas": deleted}


@app.post("/reindexar")
async def reindexar(x_api_key: Optional[str] = Header(default=None)):
    """Indexa novos arquivos da pasta rag_files sem reiniciar o servidor."""
    _check_admin_key(x_api_key)
    import asyncio
    loop = asyncio.get_running_loop()
    from src.vectorstore import reload_vectorstore
    ok, msg = await loop.run_in_executor(None, reload_vectorstore)
    logger.info("Reindexacao via endpoint: %s", msg)
    return {"sucesso": ok, "mensagem": msg}


@app.post("/webhook")
async def webhook(payload: EvolutionWebhookPayload):
    if payload.event != "messages.upsert" or not payload.data:
        return {"status": "ok"}

    data = payload.data
    key = data.key or _EvolutionKey()

    if key.fromMe:
        return {"status": "ok"}

    chat_id = key.remoteJid
    sender_name = data.pushName or ""
    msg_content = data.message or _EvolutionMessage()

    message = (
        msg_content.conversation
        or (msg_content.extendedTextMessage or {}).get("text")
    )

    if not chat_id or "@g.us" in chat_id:
        logger.debug("Mensagem de grupo ignorada (chat_id=%s).", chat_id)
        return {"status": "ok"}

    phone = chat_id.split("@")[0]

    if not is_authorized(phone):
        logger.info("Acesso negado para %s (%s)", sender_name or phone, phone)
        prefixo = f"{sender_name} não está autorizado" if sender_name else "Você não está autorizado"
        send_whatsapp_message(chat_id, f"Olá! {prefixo} a usar este assistente. Entre em contato com {SUPPORT_CONTACT}.")
        return {"status": "ok"}

    if await _is_rate_limited(phone):
        logger.warning("Rate limit atingido para %s", phone)
        send_whatsapp_message(chat_id, "Você está enviando mensagens muito rapidamente. Aguarde um momento.")
        return {"status": "ok"}

    _limpar_key = f"pending_limpar:{phone}"
    if is_admin(phone) and await _redis.exists(_limpar_key):
        await _redis.delete(_limpar_key)
        resposta_msg = (message or "").strip().lower()
        if resposta_msg in ("sim", "s", "yes"):
            deleted = clear_all_sessions()
            logger.info("[admin] /limpar confirmado por %s (%s). Sessoes removidas: %d", sender_name or phone, phone, deleted)
            reply = random.choice([
                f"Pronto! Histórico de todos os usuários foi apagado. ({deleted} sessões removidas)",
                f"Feito. {deleted} histórico(s) removido(s) com sucesso.",
                f"Ok, segui com a exclusão. {deleted} sessão(ões) apagada(s).",
            ])
        else:
            logger.info("[admin] /limpar cancelado por %s (%s)", sender_name or phone, phone)
            reply = random.choice([
                "Ok, processo cancelado. Nenhum histórico foi apagado.",
                "Tudo bem, mantive os dados como estão.",
                "Entendido, não fiz nada. Os históricos seguem intactos.",
            ])
        send_whatsapp_message(chat_id, reply)
        return {"status": "ok"}

    if message and message.startswith("/"):
        if is_admin(phone):
            # /limpar requer confirmação — intercepta antes de _handle_admin_command
            if message.strip().lower() == "/limpar":
                await _redis.setex(_limpar_key, 90, "1")
                send_whatsapp_message(
                    chat_id,
                    "Tem certeza que deseja apagar o histórico de *todos* os usuários?\n\nResponda *SIM* para confirmar ou *NÃO* para cancelar.",
                )
                return {"status": "ok"}
            logger.info("[admin] Comando recebido de %s (%s): %s", sender_name or phone, phone, message.strip())
            response = _handle_admin_command(message.strip(), admin_phone=phone, sender_name=sender_name)
            if response:
                logger.info("[admin] Resposta do comando para %s: %.120s", phone, response)
                send_whatsapp_message(chat_id, response)
            else:
                logger.warning("[admin] Comando desconhecido de %s (%s): %s", sender_name or phone, phone, message.strip())
                send_whatsapp_message(chat_id, f"Comando não reconhecido: *{message.strip().split()[0]}*\n\nDigite */ajuda* para ver os comandos disponíveis.")
            return {"status": "ok"}
        else:
            logger.warning("[admin] Tentativa de comando por nao-admin %s (%s): %s", sender_name or phone, phone, message.strip())
            send_whatsapp_message(chat_id, "Comando não reconhecido.")
            return {"status": "ok"}

    if not message and msg_content.audioMessage:
        audio_b64 = get_media_base64(key.model_dump())
        if audio_b64:
            message = transcribe_audio(audio_b64)
            logger.info("Áudio transcrito de %s: %.80s", sender_name or chat_id, message)

    if not message and msg_content.imageMessage:
        caption = (msg_content.imageMessage or {}).get("caption") or ""
        image_b64 = get_media_base64(key.model_dump())
        if image_b64:
            image_description = extract_text_from_image(image_b64)
            logger.info("Imagem interpretada de %s: %.80s", sender_name or chat_id, image_description)
            message = f"{caption}\n\n{image_description}".strip() if caption else image_description

    if not message:
        logger.debug("Mensagem sem texto ignorada (chat_id=%s, tipo provavelmente midia/sticker/reacao).", chat_id)
        return {"status": "ok"}

    logger.info("Mensagem de %s: %.80s", sender_name or chat_id, message)

    await buffer_message(chat_id=chat_id, message=message, sender_name=sender_name)

    return {"status": "ok"}


def _param_error(uso: str) -> str:
    """Retorna uma mensagem natural e variada pedindo que o admin corrija os parâmetros."""
    templates = [
        "Parece que os parâmetros estão incorretos. Tente novamente assim:\n{uso}",
        "Não consegui identificar os parâmetros. O formato esperado é:\n{uso}",
        "Algo não está certo nos parâmetros informados. Use:\n{uso}",
        "Parâmetro ausente ou incorreto. Tente novamente:\n{uso}",
        "Não foi possível executar o comando — verifique os parâmetros e tente:\n{uso}",
    ]
    return random.choice(templates).format(uso=uso)


def _handle_admin_command(message: str, admin_phone: str, sender_name: str = "") -> str | None:
    """
    Processa comandos administrativos enviados via WhatsApp.

    Comandos disponíveis:
      /autorizar 5511999999999 ; Nome ; Cargo ; Casa [; admin]
      /bloquear 5511999999999
      /desbloquear 5511999999999
      /remover 5511999999999
      /usuarios [admin]
      /reindexar
      /ajuda
    """
    parts = message.split(None, 1)  # divide em comando + resto
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/autorizar":
        return _cmd_autorizar(args, admin_phone)

    if cmd == "/bloquear":
        phone = args.strip()
        if not phone:
            return _param_error("/bloquear 5511999999999")
        return revoke(phone, revoked_by=admin_phone)

    if cmd == "/desbloquear":
        phone = args.strip()
        if not phone:
            return _param_error("/desbloquear 5511999999999")
        return unblock(phone, unblocked_by=admin_phone)

    if cmd == "/remover":
        phone = args.strip()
        if not phone:
            return _param_error("/remover 5511999999999")
        return delete_user(phone, deleted_by=admin_phone)

    if cmd == "/atualizar":
        return _cmd_atualizar(args, admin_phone)

    if cmd == "/usuarios":
        return _cmd_usuarios(admin_only=args.strip().lower() == "admin")

    if cmd == "/historico":
        parts_h = args.strip().split()
        if not parts_h:
            return _param_error("/historico 5511999999999 [dias]")
        phone_arg = parts_h[0]
        days_arg: int | None = None
        if len(parts_h) >= 2:
            try:
                days_arg = int(parts_h[1])
                if days_arg <= 0 or days_arg > 365:
                    return _param_error("/historico 5511999999999 [dias]  — o número de dias deve ser entre 1 e 365")
            except ValueError:
                return _param_error("/historico 5511999999999 [dias]")
        return _cmd_historico(phone_arg, days=days_arg)

    if cmd == "/limpar_usuario":
        phone_arg = args.strip()
        if not phone_arg:
            return _param_error("/limpar_usuario 5511999999999")
        session_id = f"{phone_arg}@s.whatsapp.net"
        clear_session(session_id)
        logger.info("[admin] Historico de %s limpo por %s (%s)", phone_arg, sender_name or admin_phone, admin_phone)
        return f"Histórico de {phone_arg} apagado com sucesso."

    if cmd == "/reindexar":
        from src.vectorstore import reload_vectorstore
        ok, msg = reload_vectorstore()
        return msg

    if cmd == "/ajuda":
        return (
            "*Comandos disponíveis:*\n\n"
            "*/autorizar* 5511999 ; Nome ; Cargo ; Casa\n"
            "→ Autoriza novo usuario padrao\n\n"
            "*/autorizar* 5511999 ; Nome ; Cargo ; Casa ; admin\n"
            "→ Autoriza novo usuario como administrador\n\n"
            "*/bloquear* 5511999\n"
            "→ Bloqueia acesso de um usuario\n\n"
            "*/desbloquear* 5511999\n"
            "→ Desbloqueia um usuario\n\n"
            "*/remover* 5511999\n"
            "→ Remove usuario permanentemente\n\n"
            "*/atualizar* 5511999 ; 5511888\n"
            "→ Atualiza o numero de telefone de um usuario\n\n"
            "*/usuarios*\n"
            "→ Lista usuarios padrao cadastrados\n\n"
            "*/usuarios admin*\n"
            "→ Lista administradores cadastrados\n\n"
            "*/historico* 5511999 dias\n"
            "→ Exibe o historico de conversa de um usuario (todo o historico ou filtrado por dias)\n\n"
            "*/limpar_usuario* 5511999\n"
            "→ Apaga o historico de conversa de um usuario especifico\n\n"
            "*/limpar*\n"
            "→ Apaga o historico de conversa de todos os usuarios\n\n"
            "*/reindexar*\n"
            "→ Indexa novos arquivos da pasta rag_files sem reiniciar\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "*Bases de dados disponíveis:*\n\n"
            "📊 *Vendas* — faturamento, ticket médio, fluxo de pessoas, produtos, funcionários, descontos\n\n"
            "🛵 *Delivery* — pedidos e faturamento por plataforma (iFood, Rappi, app próprio)\n\n"
            "↩️ *Estornos* — cancelamentos, devoluções e motivos por produto/funcionário\n\n"
            "🎯 *Metas* — realizado vs orçado, atingimento e delta por casa\n\n"
            "💳 *Formas de pagamento* — receita por método (PIX, cartão, dinheiro, etc.)\n\n"
            "🎁 *Cortesias* — itens cortesia por produto, funcionario, tipo e casa\n\n"
            "🛒 *Compras* — pedidos de compra, fornecedores e notas fiscais de entrada\n\n"
            "📄 *Documentos internos* — políticas, procedimentos, organograma e contatos"
        )

    return None  # comando desconhecido — segue fluxo normal do bot


def _cmd_autorizar(args: str, admin_phone: str) -> str:
    """
    Formato: 5511999999999 ; Nome ; Cargo ; Casa
         ou: 5511999999999 ; Nome ; Cargo ; Casa ; admin
    """
    fields = [f.strip() for f in args.split(";")]

    if len(fields) < 4:
        return _param_error("/autorizar 5511999999999 ; Nome ; Cargo ; Casa")

    phone_part = fields[0]
    nome  = fields[1]
    cargo = fields[2]
    casa  = fields[3]
    admin = len(fields) >= 5 and fields[4].lower() == "admin"
    added_by_nome = get_user_nome(admin_phone)

    return authorize(
        phone=phone_part,
        nome=nome,
        cargo=cargo,
        casa=casa,
        added_by_tel=admin_phone,
        added_by_nome=added_by_nome,
        admin=admin,
    )


def _cmd_atualizar(args: str, admin_phone: str) -> str:
    """
    Formato: /atualizar 5511999999999 ; 5511888888888
    """
    fields = [f.strip() for f in args.split(";")]
    if len(fields) < 2 or not fields[0] or not fields[1]:
        return _param_error("/atualizar 5511999999999 ; 5511888888888\n(numero atual ; numero novo)")
    return update_phone(fields[0], fields[1], updated_by=admin_phone)


def _cmd_historico(phone: str, days: int | None = None) -> str:
    """Retorna o histórico de conversa de um usuário.

    Se ``days`` for fornecido, filtra apenas mensagens dos últimos N dias.
    Caso contrário, retorna todo o histórico disponível (até 10 dias).
    """
    import time

    since_ts: float | None = None
    if days is not None:
        since_ts = time.time() - days * 86400

    session_id = f"{phone}@s.whatsapp.net"
    messages = get_session_messages(session_id, since_ts=since_ts)

    periodo = f"últimos {days} dia(s)" if days is not None else "todo o histórico"
    if not messages:
        return f"Nenhum histórico encontrado para {phone} ({periodo})."

    nome = get_user_nome(phone) or phone
    lines = [f"*Histórico de {nome} ({phone}) — {periodo}:*"]
    for msg in messages:
        is_human = msg["role"] in ("human", "HumanMessage")
        prefix = "👤 *Usuário:*" if is_human else "🤖 *Assistente:*"
        content = msg["content"].strip()
        # Trunca respostas longas na última linha completa dentro do limite
        if len(content) > 300:
            truncated = content[:297]
            last_newline = truncated.rfind("\n")
            content = (truncated[:last_newline] if last_newline > 100 else truncated) + "\n_(truncado)_"
        lines.append(f"\n{prefix}\n{content}")

    return "\n".join(lines)


def _cmd_usuarios(admin_only: bool = False) -> str:
    users = list_users()
    if not users:
        return "Nenhum usuário cadastrado."

    filtered = [u for u in users if bool(u["is_admin"]) == admin_only]
    ativos   = [u for u in filtered if u["active"]]
    inativos = [u for u in filtered if not u["active"]]

    if not ativos and not inativos:
        return "Nenhum administrador cadastrado." if admin_only else "Nenhum usuário padrão cadastrado."

    titulo = "*Administradores:*" if admin_only else "*Usuários padrão:*"
    lines = [titulo]
    for u in ativos:
        linha = f"• {u['telefone']} | {u['nome']} | {u['cargo']} | {u['casa']}"
        if admin_only:
            linha += " | _admin_"
        lines.append(linha)

    if inativos:
        lines.append("\n*Bloqueados:*")
        for u in inativos:
            lines.append(f"• {u['nome']} ({u['telefone']})")

    return "\n".join(lines)
