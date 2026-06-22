import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mocka src.chains antes de importar message_buffer para evitar imports pesados
sys.modules.setdefault("src.chains", MagicMock())

import src.message_buffer as mb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(done: bool = False, cancelled: bool = False) -> MagicMock:
    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = done
    task.cancel = MagicMock()
    return task


# ---------------------------------------------------------------------------
# _is_cancel_command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word", ["cancela", "cancel", "cancelar", "pare", "parar", "stop"])
def test_is_cancel_command_palavras_validas(word):
    assert mb._is_cancel_command(word) is True


@pytest.mark.parametrize("word", ["CANCELA", "  pare  ", "STOP"])
def test_is_cancel_command_case_insensitive_e_espacos(word):
    assert mb._is_cancel_command(word) is True


@pytest.mark.parametrize("word", ["cancelar tudo", "preciso de dados", ""])
def test_is_cancel_command_nao_reconhece_frases(word):
    assert mb._is_cancel_command(word) is False


def test_is_cancel_command_frase_reconhece_cancela_isso():
    assert mb._is_cancel_command("cancela isso") is True


# ---------------------------------------------------------------------------
# buffer_message — fluxo de cancelamento
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cancel_cancela_task_existente_e_limpa_buffer():
    chat_id = "5511999@s.whatsapp.net"
    task = _make_task(done=False)
    mb.debounce_tasks[chat_id] = task

    with patch.object(mb.redis_client, "delete", new_callable=AsyncMock) as mock_del, \
         patch.object(mb.redis_client, "rpush", new_callable=AsyncMock), \
         patch.object(mb.redis_client, "expire", new_callable=AsyncMock), \
         patch("src.message_buffer.send_whatsapp_message") as mock_send:

        await mb.buffer_message(chat_id=chat_id, message="cancela", sender_name="User")

    task.cancel.assert_called_once()
    from src.config import BUFFER_KEY_SUFIX
    mock_del.assert_called_once_with(f"{chat_id}{BUFFER_KEY_SUFIX}")
    mock_send.assert_called_once()
    sent_text = mock_send.call_args.kwargs["text"]
    assert sent_text in mb._CANCEL_RESPONSES
    assert chat_id not in mb.debounce_tasks


@pytest.mark.anyio
async def test_cancel_sem_task_ativa_nao_levanta_excecao():
    chat_id = "5511888@s.whatsapp.net"
    mb.debounce_tasks.pop(chat_id, None)

    with patch.object(mb.redis_client, "delete", new_callable=AsyncMock), \
         patch("src.message_buffer.send_whatsapp_message"):

        await mb.buffer_message(chat_id=chat_id, message="pare", sender_name="User")


@pytest.mark.anyio
async def test_cancel_nao_cria_nova_debounce_task():
    chat_id = "5511777@s.whatsapp.net"
    mb.debounce_tasks.pop(chat_id, None)

    with patch.object(mb.redis_client, "delete", new_callable=AsyncMock), \
         patch("src.message_buffer.send_whatsapp_message"), \
         patch("asyncio.create_task") as mock_create:

        await mb.buffer_message(chat_id=chat_id, message="stop", sender_name="User")

    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# buffer_message — fluxo normal (mensagem comum)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_mensagem_normal_adiciona_ao_buffer_e_cria_task():
    chat_id = "5511666@s.whatsapp.net"
    mb.debounce_tasks.pop(chat_id, None)

    fake_task = MagicMock(spec=asyncio.Task)

    with patch.object(mb.redis_client, "rpush", new_callable=AsyncMock) as mock_push, \
         patch.object(mb.redis_client, "expire", new_callable=AsyncMock), \
         patch("src.message_buffer.handle_debounce", new=MagicMock()), \
         patch("asyncio.create_task", return_value=fake_task) as mock_create:

        await mb.buffer_message(chat_id=chat_id, message="qual o faturamento?", sender_name="User")

    mock_push.assert_called_once()
    mock_create.assert_called_once()
    assert mb.debounce_tasks[chat_id] is fake_task


@pytest.mark.anyio
async def test_mensagem_normal_reseta_debounce_existente():
    chat_id = "5511555@s.whatsapp.net"
    old_task = _make_task(done=False)
    mb.debounce_tasks[chat_id] = old_task

    fake_task = MagicMock(spec=asyncio.Task)

    with patch.object(mb.redis_client, "rpush", new_callable=AsyncMock), \
         patch.object(mb.redis_client, "expire", new_callable=AsyncMock), \
         patch("src.message_buffer.handle_debounce", new=MagicMock()), \
         patch("asyncio.create_task", return_value=fake_task):

        await mb.buffer_message(chat_id=chat_id, message="novo pedido", sender_name="User")

    old_task.cancel.assert_called_once()
    assert mb.debounce_tasks[chat_id] is fake_task
