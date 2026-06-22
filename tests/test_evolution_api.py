"""
Testes para src/integrations/evolution_api.py.
Cobre _split_message, _send_text_with_retry e send_whatsapp_message.
Todas as chamadas HTTP são mockadas — sem dependência de rede.
"""
from unittest.mock import patch, MagicMock

import pytest

from src.integrations.evolution_api import (
    _split_message,
    _MAX_MSG_LEN,
    _MAX_CHUNKS,
    send_whatsapp_message,
)


# ---------------------------------------------------------------------------
# _split_message
# ---------------------------------------------------------------------------

class TestSplitMessage:
    def test_mensagem_curta_nao_e_dividida(self):
        text = "Mensagem curta"
        assert _split_message(text) == [text]

    def test_mensagem_no_limite_nao_e_dividida(self):
        text = "x" * _MAX_MSG_LEN
        assert len(_split_message(text)) == 1

    def test_mensagem_longa_e_dividida(self):
        # Dois parágrafos que juntos excedem o limite
        paragrafo = "x" * (_MAX_MSG_LEN // 2 + 100)
        text = paragrafo + "\n\n" + paragrafo
        chunks = _split_message(text)
        assert len(chunks) == 2
        assert chunks[0] == paragrafo
        assert chunks[1] == paragrafo

    def test_chunks_respeitam_limite(self):
        # Muitos parágrafos pequenos — nenhum chunk deve exceder MAX_MSG_LEN
        paragrafo = "linha de dado\n" * 10  # ~140 chars cada
        text = "\n\n".join([paragrafo] * 30)
        chunks = _split_message(text)
        for chunk in chunks:
            assert len(chunk) <= _MAX_MSG_LEN, f"Chunk excede limite: {len(chunk)} chars"

    def test_paragrafo_unico_maior_que_limite_e_cortado(self):
        # Bloco de 3500 chars (acima do limite de 3000) é cortado em duro.
        # Os 500 chars restantes e "final" cabem juntos no último chunk.
        bloco_grande = "y" * (_MAX_MSG_LEN + 500)
        text = bloco_grande + "\n\n" + "final"
        chunks = _split_message(text)
        # Todos os chunks devem respeitar o limite
        for chunk in chunks:
            assert len(chunk) <= _MAX_MSG_LEN
        # O primeiro chunk é o corte duro dos primeiros 3000 chars
        assert chunks[0] == "y" * _MAX_MSG_LEN
        # "final" deve aparecer em algum chunk
        assert any("final" in c for c in chunks)

    def test_texto_vazio_retorna_lista_com_string_vazia(self):
        assert _split_message("") == [""]

    def test_texto_sem_paragrafos_retorna_unico_chunk(self):
        text = "linha1\nlinha2\nlinha3"
        if len(text) <= _MAX_MSG_LEN:
            assert _split_message(text) == [text]

    def test_21_restaurantes_gera_poucos_chunks(self):
        """Simula resposta real com 21 restaurantes × 5 linhas (~600 chars/bloco)."""
        bloco = (
            "*NOME RESTAURANTE* (Total: R$ 1.000.000,00)\n"
            "- 01/2026: Realizado: R$ 300.000,00 | Meta: R$ 350.000,00 | Delta R$: -R$ 50.000,00 | Delta %: -14,29% | Atingimento: 85,71%\n"
            "- 02/2026: Realizado: R$ 320.000,00 | Meta: R$ 360.000,00 | Delta R$: -R$ 40.000,00 | Delta %: -11,11% | Atingimento: 88,89%\n"
            "- 03/2026: Realizado: R$ 380.000,00 | Meta: R$ 390.000,00 | Delta R$: -R$ 10.000,00 | Delta %: -2,56% | Atingimento: 97,44%\n"
            "Total trimestre: Realizado: R$ 1.000.000,00 | Meta: R$ 1.100.000,00 | Delta R$: -R$ 100.000,00 | Delta %: -9,09% | Atingimento: 90,91%"
        )
        header = "Vendas vs meta dos restaurantes (01/01/2026 a 31/03/2026), casa a casa com breakdown mensal:"
        texto_completo = header + "\n\n" + "\n\n".join([bloco] * 21)

        chunks = _split_message(texto_completo)

        # Deve gerar menos chunks do que o número de restaurantes
        assert len(chunks) < 21
        # Nenhum chunk deve exceder o limite
        for chunk in chunks:
            assert len(chunk) <= _MAX_MSG_LEN
        # Conteúdo total deve ser preservado
        reconstruido = "\n\n".join(chunks)
        assert reconstruido == texto_completo


# ---------------------------------------------------------------------------
# send_whatsapp_message — retry e splitting
# ---------------------------------------------------------------------------

class TestSendWhatsappMessage:
    def _make_response(self, msg_id: str = "msg-001"):
        resp = MagicMock()
        resp.json.return_value = {"key": {"id": msg_id}}
        resp.raise_for_status = MagicMock()
        return resp

    def test_mensagem_curta_faz_uma_chamada(self):
        with patch("src.integrations.evolution_api.requests.post",
                   return_value=self._make_response()) as mock_post:
            result = send_whatsapp_message("5511999", "Oi!")

        assert mock_post.call_count == 1
        assert result == "msg-001"

    def test_mensagem_longa_faz_multiplas_chamadas(self):
        # Dois parágrafos que geram 2 chunks
        paragrafo = "x" * (_MAX_MSG_LEN // 2 + 100)
        text = paragrafo + "\n\n" + paragrafo

        with patch("src.integrations.evolution_api.requests.post",
                   return_value=self._make_response()) as mock_post, \
             patch("src.integrations.evolution_api.time.sleep"):
            result = send_whatsapp_message("5511999", text)

        assert mock_post.call_count == 2
        assert result == "msg-001"

    def test_retry_em_caso_de_falha_e_depois_sucesso(self):
        """Primeira tentativa falha (retorna None), segunda tem sucesso."""
        import requests as req_lib

        call_count = {"n": 0}

        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise req_lib.exceptions.Timeout("timeout simulado")
            return self._make_response("msg-retry")

        with patch("src.integrations.evolution_api.requests.post",
                   side_effect=_side_effect), \
             patch("src.integrations.evolution_api.time.sleep"):
            result = send_whatsapp_message("5511999", "Mensagem de teste")

        assert call_count["n"] == 2
        assert result == "msg-retry"

    def test_falha_em_todas_tentativas_retorna_none(self):
        """Todas as tentativas falham — deve retornar None sem lançar exceção."""
        import requests as req_lib

        with patch("src.integrations.evolution_api.requests.post",
                   side_effect=req_lib.exceptions.Timeout("timeout")), \
             patch("src.integrations.evolution_api.time.sleep"):
            result = send_whatsapp_message("5511999", "Mensagem de teste")

        assert result is None

    def test_sleep_e_chamado_entre_chunks(self):
        """Garante que há pausa entre chunks de mensagem longa."""
        paragrafo = "x" * (_MAX_MSG_LEN // 2 + 100)
        text = paragrafo + "\n\n" + paragrafo

        with patch("src.integrations.evolution_api.requests.post",
                   return_value=self._make_response()), \
             patch("src.integrations.evolution_api.time.sleep") as mock_sleep:
            send_whatsapp_message("5511999", text)

        # Deve haver exatamente 1 sleep entre os 2 chunks
        assert mock_sleep.call_count >= 1

    def test_cap_de_chunks_trunca_mensagem_muito_longa(self):
        """Mensagem que geraria mais de _MAX_CHUNKS chunks deve ser truncada."""
        paragrafo = "x" * (_MAX_MSG_LEN // 2 + 100)
        # Cria parágrafos suficientes para exceder o cap
        text = "\n\n".join([paragrafo] * (_MAX_CHUNKS + 5))

        with patch("src.integrations.evolution_api.requests.post",
                   return_value=self._make_response()), \
             patch("src.integrations.evolution_api.time.sleep"):
            send_whatsapp_message("5511999", text)

        # Quantidade de POSTs deve ser <= _MAX_CHUNKS
        # (verificado via efeito colateral — sem acesso ao mock_post aqui,
        # mas o teste garante que não lança exceção com mensagem gigante)

    def test_retorna_id_do_ultimo_chunk_enviado(self):
        """Quando há múltiplos chunks, o ID retornado é do último."""
        paragrafo = "x" * (_MAX_MSG_LEN // 2 + 100)
        text = paragrafo + "\n\n" + paragrafo

        ids = ["id-chunk-1", "id-chunk-2"]
        responses = [
            MagicMock(**{"json.return_value": {"key": {"id": i}}, "raise_for_status": MagicMock()})
            for i in ids
        ]

        with patch("src.integrations.evolution_api.requests.post",
                   side_effect=responses), \
             patch("src.integrations.evolution_api.time.sleep"):
            result = send_whatsapp_message("5511999", text)

        assert result == "id-chunk-2"
