import asyncio
import logging
import random
import re

import redis.asyncio as redis

from src.config import REDIS_URL, BUFFER_KEY_SUFIX, DEBOUNCE_SECONDS, BUFFER_TTL
from src.integrations.evolution_api import (
    send_whatsapp_message, send_whatsapp_image, send_whatsapp_document,
    send_whatsapp_presence,
)
from src.chains import route_and_invoke, generate_thinking_message

_CHART_RE = re.compile(r'\[CHART:(chart:[a-f0-9]+)\|caption:([^\]]*)\]')
_EXCEL_RE = re.compile(r'\[EXCEL:(excel:[a-f0-9]+)\|caption:([^\]]*)\]')

logger = logging.getLogger(__name__)

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
debounce_tasks: dict[str, asyncio.Task] = {}
_debounce_lock = asyncio.Lock()

_CANCEL_EXACT = {
    "cancela", "cancel", "cancelar", "pare", "parar", "stop",
    "esquece", "não quero", "nao quero", "chega", "desiste", "desistir",
}
_CANCEL_PHRASES = {
    "não quero mais", "nao quero mais", "esquece isso", "deixa pra lá",
    "deixa pra la", "para tudo", "cancela isso", "cancela tudo",
    "para isso", "esquece a pergunta",
}
_CANCEL_RESPONSES = [
    "Ok, solicitação cancelada.",
    "Tudo bem, cancelei por aqui.",
    "Certo, processo interrompido.",
    "Ok, deixa pra lá então.",
    "Entendido, cancelei.",
]


def _is_cancel_command(message: str) -> bool:
    msg = message.strip().lower()
    if msg in _CANCEL_EXACT:
        return True
    return any(phrase in msg for phrase in _CANCEL_PHRASES)


async def _deliver_media(
    loop,
    chat_id: str,
    redis_key: str,
    label: str,
    send_fn,
    text_response: str,
) -> None:
    """Envia texto opcional + arquivo de mídia (gráfico ou Excel) para o WhatsApp."""
    if text_response:
        await loop.run_in_executor(
            None, lambda: send_whatsapp_message(number=chat_id, text=text_response)
        )
    b64 = await redis_client.get(redis_key)
    if b64:
        await redis_client.delete(redis_key)
        await loop.run_in_executor(None, lambda: send_fn(b64))
        logger.info("[buffer] %s enviado com sucesso para %s.", label, chat_id)
    else:
        logger.warning("[buffer] Chave do %s nao encontrada no Redis: %s", label, redis_key)


async def buffer_message(chat_id: str, message: str, sender_name: str = "") -> None:
    buffer_key = f"{chat_id}{BUFFER_KEY_SUFIX}"

    if _is_cancel_command(message):
        async with _debounce_lock:
            existing = debounce_tasks.get(chat_id)
            if existing and not existing.done():
                existing.cancel()
                logger.info("[buffer] Tarefa de %s cancelada pelo usuario.", chat_id)
            debounce_tasks.pop(chat_id, None)
        await redis_client.delete(buffer_key)
        loop = asyncio.get_running_loop()
        cancel_msg = random.choice(_CANCEL_RESPONSES)
        await loop.run_in_executor(
            None, lambda: send_whatsapp_message(number=chat_id, text=cancel_msg)
        )
        return

    await redis_client.rpush(buffer_key, message)
    await redis_client.expire(buffer_key, BUFFER_TTL)

    logger.info("[buffer] Mensagem adicionada ao buffer de %s (len=%d): %.80s", chat_id, len(message), message)

    async with _debounce_lock:
        existing = debounce_tasks.get(chat_id)
        if existing and not existing.done():
            existing.cancel()
            logger.debug("[buffer] Debounce resetado para %s — nova mensagem chegou antes do timer.", chat_id)
        debounce_tasks[chat_id] = asyncio.create_task(handle_debounce(chat_id, sender_name))


async def _keep_typing(chat_id: str, loop, stop_event: asyncio.Event) -> None:
    """Envia 'digitando...' a cada 3s até stop_event ser setado."""
    while not stop_event.is_set():
        await loop.run_in_executor(None, lambda: send_whatsapp_presence(number=chat_id))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=3)
        except asyncio.TimeoutError:
            pass


async def handle_debounce(chat_id: str, sender_name: str = "") -> None:
    try:
        await asyncio.sleep(DEBOUNCE_SECONDS)

        buffer_key = f"{chat_id}{BUFFER_KEY_SUFIX}"
        messages = await redis_client.lrange(buffer_key, 0, -1)

        full_message = "\n".join(messages).strip()
        if full_message:
            logger.info("Enviando mensagem agrupada para %s: %s", chat_id, full_message)

            loop = asyncio.get_running_loop()

            stop_typing = asyncio.Event()
            typing_task = asyncio.create_task(_keep_typing(chat_id, loop, stop_typing))

            def _on_thinking():
                thinking_msg = generate_thinking_message(full_message)
                send_whatsapp_message(number=chat_id, text=thinking_msg)

            try:
                ai_response = await loop.run_in_executor(
                    None,
                    lambda: route_and_invoke(
                        message=full_message,
                        session_id=chat_id,
                        sender_name=sender_name,
                        on_thinking=_on_thinking,
                    ),
                )
            finally:
                stop_typing.set()
                await typing_task

            logger.info("Resposta do agente para %s: %.500s", chat_id, ai_response)

            chart_match = _CHART_RE.search(ai_response)
            excel_match = _EXCEL_RE.search(ai_response)

            if chart_match:
                chart_key = chart_match.group(1)
                caption = chart_match.group(2)
                logger.info("[buffer] Resposta contem grafico (key=%s, caption='%s').", chart_key, caption)
                text_response = _CHART_RE.sub('', ai_response).strip()
                await _deliver_media(
                    loop, chat_id, chart_key, "grafico",
                    lambda b64: send_whatsapp_image(number=chat_id, b64=b64, caption=caption),
                    text_response,
                )
            elif excel_match:
                excel_key = excel_match.group(1)
                filename = excel_match.group(2)
                logger.info("[buffer] Resposta contem Excel (key=%s, filename='%s').", excel_key, filename)
                text_response = _EXCEL_RE.sub('', ai_response).strip()
                await _deliver_media(
                    loop, chat_id, excel_key, f"Excel '{filename}'",
                    lambda b64: send_whatsapp_document(number=chat_id, b64=b64, filename=filename),
                    text_response,
                )
            else:
                logger.info("[buffer] Enviando resposta em texto para %s.", chat_id)
                await loop.run_in_executor(
                    None, lambda: send_whatsapp_message(number=chat_id, text=ai_response)
                )


        await redis_client.delete(buffer_key)

    except asyncio.CancelledError:
        logger.debug("[buffer] Debounce cancelado para %s (nova mensagem ou comando cancel).", chat_id)

    except Exception:
        logger.exception("[buffer] Erro inesperado no processamento de %s — task encerrada.", chat_id)

    finally:
        async with _debounce_lock:
            if debounce_tasks.get(chat_id) is asyncio.current_task():
                debounce_tasks.pop(chat_id, None)
