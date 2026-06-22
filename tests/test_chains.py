from contextlib import ExitStack
from datetime import datetime as _real_datetime
from unittest.mock import patch, MagicMock

from src.chains import _complete_dates, _classify_intent, route_and_invoke


def _freeze(year: int, month: int, day: int):
    """Congela datetime.now() para uma data específica dentro de src.chains."""
    mock = patch("src.chains.datetime")

    class _Ctx:
        def __enter__(self):
            self._patcher = mock.__enter__()
            self._patcher.now.return_value = _real_datetime(year, month, day, 10, 0)
            self._patcher.side_effect = lambda *a, **kw: _real_datetime(*a, **kw)
            return self._patcher

        def __exit__(self, *args):
            mock.__exit__(*args)

    return _Ctx()


def _make_mock_model(content: str):
    return type("M", (), {"invoke": lambda self, x: type("R", (), {"content": content})()})()


def _mock_model_ctx(content: str):
    """Mocka _get_model E _get_fallback_model para evitar chamadas reais ao LLM."""
    mock_obj = _make_mock_model(content)

    class _Ctx:
        def __enter__(self_):
            self_._stack = ExitStack()
            self_._stack.enter_context(patch("src.chains._get_model", return_value=mock_obj))
            self_._stack.enter_context(patch("src.chains._get_fallback_model", return_value=None))
            return mock_obj

        def __exit__(self_, *args):
            self_._stack.__exit__(*args)

    return _Ctx()


def _make_history_with_messages():
    """Retorna mock de histórico com uma mensagem (usuário não é primeiro contato)."""
    fake_msg = MagicMock()
    fake_msg.type = "human"
    history = MagicMock()
    history.messages = [fake_msg]
    return history


class TestCompleteDates:
    def test_data_com_ano_nao_muda(self):
        with _freeze(2025, 6, 15):
            assert _complete_dates("vendas de 10/03/2025") == "vendas de 10/03/2025"

    def test_data_sem_ano_recebe_ano_atual(self):
        with _freeze(2025, 6, 15):
            assert _complete_dates("vendas de 10/03") == "vendas de 10/03/2025"

    def test_data_virada_de_ano_usa_ano_anterior(self):
        # 02/jan consultando 28/dez — deve completar com ano anterior
        with _freeze(2026, 1, 2):
            assert _complete_dates("vendas de 28/12") == "vendas de 28/12/2025"

    def test_data_futuro_proximo_mantem_ano_atual(self):
        # 12/mar consultando 05/abr (24 dias à frente) — ainda ano atual
        with _freeze(2025, 3, 12):
            assert _complete_dates("vendas até 05/04") == "vendas até 05/04/2025"

    def test_ano_com_digitos_extras_e_truncado(self):
        with _freeze(2025, 6, 15):
            assert _complete_dates("data 10/03/202599") == "data 10/03/2025"

    def test_mensagem_sem_data_nao_muda(self):
        with _freeze(2025, 6, 15):
            msg = "qual o total de vendas?"
            assert _complete_dates(msg) == msg

    def test_multiplas_datas_na_mensagem(self):
        with _freeze(2025, 6, 15):
            result = _complete_dates("de 01/03 até 31/03")
            assert result == "de 01/03/2025 até 31/03/2025"


class TestClassifyIntent:
    def test_retorna_sql(self):
        with _mock_model_ctx("sql"):
            assert _classify_intent("total de vendas hoje") == "sql"

    def test_retorna_docs(self):
        with _mock_model_ctx("docs"):
            assert _classify_intent("qual a política de férias") == "docs"

    def test_retorna_ambos(self):
        with _mock_model_ctx("ambos"):
            assert _classify_intent("vendas e política") == "ambos"

    def test_retorna_geral(self):
        with _mock_model_ctx("geral"):
            assert _classify_intent("olá, tudo bem?") == "geral"

    def test_categoria_invalida_retorna_geral(self):
        # Quando o modelo retorna categoria desconhecida o fallback é "geral"
        with _mock_model_ctx("invalido"):
            assert _classify_intent("algo") == "geral"

    def test_excecao_no_modelo_retorna_sql(self):
        # Quando o modelo lança exceção, o fallback é "sql" (mais seguro que "geral")
        with patch("src.chains._get_model") as m, \
             patch("src.chains._get_fallback_model", return_value=None):
            m.return_value.invoke.side_effect = Exception("API error")
            assert _classify_intent("algo") == "sql"


class TestGreetingBehavior:
    """Testa o comportamento de saudações no route_and_invoke."""

    def _patch_base(self):
        return (
            patch("src.chains._save_to_history"),
            patch("src.chains._metric_inc"),
        )

    def _patch_history(self):
        """Simula sessão com histórico existente — usuário retornando, sem primeiro contato."""
        return patch("src.chains.get_session_history",
                     return_value=_make_history_with_messages())

    def test_saudacao_chama_run_general_response(self):
        """Saudação de usuário retornando chama _run_general_response."""
        p1, p2 = self._patch_base()
        with p1, p2, self._patch_history(), \
             patch("src.chains._run_general_response", return_value="Ola!") as mock_llm:
            route_and_invoke("oi", session_id="123", sender_name="Railan")

        mock_llm.assert_called_once()

    def test_saudacao_nao_chama_classify_intent(self):
        """Saudação não passa pelo router — _classify_intent não deve ser chamado."""
        p1, p2 = self._patch_base()
        with p1, p2, self._patch_history(), \
             patch("src.chains._run_general_response", return_value="Ola!"), \
             patch("src.chains._classify_intent") as mock_router:
            route_and_invoke("oi", session_id="123", sender_name="Railan")

        mock_router.assert_not_called()

    def test_saudacao_retorna_resposta_do_llm(self):
        """Saudação retorna a resposta de _run_general_response (sem emojis)."""
        p1, p2 = self._patch_base()
        with p1, p2, self._patch_history(), \
             patch("src.chains._run_general_response", return_value="Ola, tudo bem!"):
            result = route_and_invoke("olá", session_id="123", sender_name="")

        assert result == "Ola, tudo bem!"

    def test_saudacao_com_emoji_e_removido(self):
        """Emojis na resposta de saudação são removidos pelo _strip_emojis."""
        p1, p2 = self._patch_base()
        with p1, p2, self._patch_history(), \
             patch("src.chains._run_general_response", return_value="Ola! 😊 Como posso ajudar?"):
            result = route_and_invoke("oi", session_id="123", sender_name="")

        assert "😊" not in result

    def test_pergunta_direta_nao_e_saudacao_vai_para_agente(self):
        """Pergunta de dados não é saudação — deve passar pelo router e ir ao agente SQL."""
        # Histórico com mensagem prévia → não é primeiro contato → sem prefix de intro
        history_with_msgs = _make_history_with_messages()
        p1, p2 = self._patch_base()
        with p1, p2, \
             patch("src.chains._cache_get", return_value=None), \
             patch("src.chains.get_session_history", return_value=history_with_msgs), \
             patch("src.chains._classify_intent", return_value="sql"), \
             patch("src.chains._run_sql_agent", return_value="resultado sql") as mock_sql, \
             patch("src.chains._cache_set"):
            result = route_and_invoke("qual o faturamento de hoje?", session_id="123", sender_name="Railan")

        mock_sql.assert_called_once()
        assert result == "resultado sql"
