import logging
import time

import requests

from src.config import (
    EVOLUTION_API_URL,
    EVOLUTION_INSTANCE_NAME,
    EVOLUTION_AUTHENTICATION_API_KEY,
)

logger = logging.getLogger(__name__)

_HEADERS = {
    "apikey": EVOLUTION_AUTHENTICATION_API_KEY,
    "Content-Type": "application/json",
}


def _send_media(number: str, mediatype: str, mimetype: str, b64: str,
                caption: str = "", filename: str = "") -> None:
    """Envia mídia (imagem ou documento) via Evolution API."""
    url = f"{EVOLUTION_API_URL}/message/sendMedia/{EVOLUTION_INSTANCE_NAME}"
    payload = {
        "number": number,
        "mediatype": mediatype,
        "mimetype": mimetype,
        "media": b64,
        "caption": caption,
    }
    if filename:
        payload["fileName"] = filename
    try:
        response = requests.post(url=url, json=payload, headers=_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Timeout ao enviar %s para %s", mediatype, number)
    except requests.exceptions.HTTPError as e:
        logger.error("Erro HTTP ao enviar %s para %s: %s — %s", mediatype, number, e.response.status_code, e.response.text)
    except requests.exceptions.RequestException as e:
        logger.error("Falha ao enviar %s para %s: %s", mediatype, number, e)


_MAX_MSG_LEN = 3000
_MAX_CHUNKS = 30  # limite de partes — resumos por vertical podem gerar ~15 chunks
_CHUNK_DELAY = 2.0
_SEND_RETRIES = 3
_RETRY_DELAY = 3.0


def _split_message(text: str, max_len: int = _MAX_MSG_LEN) -> list[str]:
    """Divide texto em chunks garantindo que nenhum exceda max_len.

    Tenta preservar parágrafos (\n\n), depois linhas (\n), e por último
    faz corte duro por caracteres — nessa ordem de preferência.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append("\n\n".join(buf))
            buf.clear()

    def push_para(para: str) -> None:
        # para é garantidamente <= max_len por quem chama
        sep = 2 if buf else 0
        if buf and len("\n\n".join(buf)) + sep + len(para) > max_len:
            flush()
        buf.append(para)

    for paragraph in text.split("\n\n"):
        if len(paragraph) <= max_len:
            push_para(paragraph)
            continue

        # Parágrafo maior que max_len — quebra por linhas
        line_buf: list[str] = []
        for line in paragraph.split("\n"):
            if len(line) > max_len:
                # Linha sozinha ultrapassa o limite — flush e corte duro
                if line_buf:
                    push_para("\n".join(line_buf))
                    line_buf = []
                for i in range(0, len(line), max_len):
                    push_para(line[i:i + max_len])
            else:
                sep = 1 if line_buf else 0
                if line_buf and len("\n".join(line_buf)) + sep + len(line) > max_len:
                    push_para("\n".join(line_buf))
                    line_buf = [line]
                else:
                    line_buf.append(line)
        if line_buf:
            push_para("\n".join(line_buf))

    flush()
    return chunks


def _send_text(number: str, text: str) -> str | None:
    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE_NAME}"
    payload = {"number": number, "text": text}
    try:
        response = requests.post(url=url, json=payload, headers=_HEADERS, timeout=15)
        response.raise_for_status()
        return response.json().get("key", {}).get("id")
    except requests.exceptions.Timeout:
        logger.error("Timeout ao enviar mensagem para %s", number)
    except requests.exceptions.HTTPError as e:
        logger.error("Erro HTTP ao enviar para %s: %s — %s", number, e.response.status_code, e.response.text)
    except requests.exceptions.RequestException as e:
        logger.error("Falha ao enviar mensagem para %s: %s", number, e)
    return None


def _send_text_with_retry(number: str, text: str) -> str | None:
    """Tenta enviar um chunk de texto com até _SEND_RETRIES tentativas."""
    for attempt in range(1, _SEND_RETRIES + 1):
        msg_id = _send_text(number, text)
        if msg_id:
            return msg_id
        if attempt < _SEND_RETRIES:
            logger.warning("Tentativa %d/%d falhou para %s — aguardando %.1fs antes de retry.",
                           attempt, _SEND_RETRIES, number, _RETRY_DELAY)
            time.sleep(_RETRY_DELAY)
    return None


def send_whatsapp_message(number: str, text: str) -> str | None:
    """
    Envia mensagem de texto via Evolution API.
    Divide automaticamente mensagens longas em múltiplas partes (cap: _MAX_CHUNKS).
    Retorna o message ID da última parte enviada com sucesso, None caso contrário.
    """
    chunks = _split_message(text)
    if len(chunks) > _MAX_CHUNKS:
        logger.warning("Mensagem para %s gerou %d chunks (acima do cap=%d) — truncando. Sugira exportar em Excel.", number, len(chunks), _MAX_CHUNKS)
        chunks = chunks[:_MAX_CHUNKS]
        chunks[-1] = chunks[-1] + "\n\n_(Resposta truncada — exporte em Excel para ver todos os dados)_"
    if len(chunks) > 1:
        logger.info("Mensagem longa para %s dividida em %d partes.", number, len(chunks))

    last_id = None
    for i, chunk in enumerate(chunks, 1):
        msg_id = _send_text_with_retry(number, chunk)
        if msg_id:
            last_id = msg_id
        elif len(chunks) > 1:
            logger.warning("Falha definitiva ao enviar parte %d/%d para %s após %d tentativas.",
                           i, len(chunks), number, _SEND_RETRIES)
        if i < len(chunks):
            time.sleep(_CHUNK_DELAY)
    return last_id


def send_whatsapp_image(number: str, b64: str, caption: str = "") -> None:
    """Envia imagem PNG via Evolution API."""
    _send_media(number, "image", "image/png", b64, caption=caption)


def send_whatsapp_document(number: str, b64: str, filename: str) -> None:
    """Envia arquivo .xlsx via Evolution API."""
    _send_media(
        number, "document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        b64, caption=filename, filename=filename,
    )


def send_whatsapp_presence(number: str) -> None:
    """Envia status 'digitando...' para o usuário via Evolution API."""
    url = f"{EVOLUTION_API_URL}/chat/sendPresence/{EVOLUTION_INSTANCE_NAME}"
    clean_number = number.split("@")[0]
    payload = {"number": clean_number, "delay": 1500, "presence": "composing"}
    try:
        response = requests.post(url=url, json=payload, headers=_HEADERS, timeout=5)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.warning("Falha ao enviar presenca para %s: %s — body: %s", number, e, e.response.text)
    except Exception as e:
        logger.warning("Falha ao enviar presenca para %s: %s", number, e)


def get_media_base64(message_key: dict) -> str:
    """Baixa mídia (áudio, imagem, etc.) da Evolution API e retorna em base64."""
    url = f"{EVOLUTION_API_URL}/chat/getBase64FromMediaMessage/{EVOLUTION_INSTANCE_NAME}"
    try:
        response = requests.post(
            url=url,
            json={"message": {"key": message_key}},
            headers=_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("base64", "")
    except requests.exceptions.Timeout:
        logger.error("Timeout ao baixar mídia: %s", message_key.get("id"))
    except requests.exceptions.HTTPError as e:
        logger.error("Erro HTTP ao baixar mídia: %s — %s", e.response.status_code, e.response.text)
    except requests.exceptions.RequestException as e:
        logger.error("Falha ao baixar mídia: %s", e)
    return ""
