"""
Testes para src/tools/resumo_tool.py e src/tools/utils.py (fmt_brl/fmt_int_br).
"""
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.tools.utils import fmt_brl, fmt_int_br
from src.tools.resumo_tool import _validate_date, _parse_input, _build_where, DremioResumoTool


# ---------------------------------------------------------------------------
# fmt_brl / fmt_int_br
# ---------------------------------------------------------------------------

class TestFmtBrl:
    def test_formata_valor_simples(self):
        assert fmt_brl(1000.0) == "R$ 1.000,00"

    def test_formata_valor_com_centavos(self):
        assert fmt_brl(1234.56) == "R$ 1.234,56"

    def test_formata_zero(self):
        assert fmt_brl(0.0) == "R$ 0,00"

    def test_formata_milhar(self):
        assert fmt_brl(1_000_000.0) == "R$ 1.000.000,00"


class TestFmtIntBr:
    def test_formata_milhar(self):
        assert fmt_int_br(1000) == "1.000"

    def test_formata_milhao(self):
        assert fmt_int_br(1_000_000) == "1.000.000"

    def test_formata_zero(self):
        assert fmt_int_br(0) == "0"


# ---------------------------------------------------------------------------
# _validate_date
# ---------------------------------------------------------------------------

class TestValidateDate:
    def test_data_valida(self):
        assert _validate_date("2026-04-13") is True

    def test_data_invalida_mes_errado(self):
        assert _validate_date("2026-13-01") is False

    def test_data_invalida_dia_errado(self):
        assert _validate_date("2026-01-32") is False

    def test_formato_errado(self):
        assert _validate_date("13/04/2026") is False

    def test_string_vazia(self):
        assert _validate_date("") is False

    def test_texto_aleatorio(self):
        assert _validate_date("ontem") is False


# ---------------------------------------------------------------------------
# _parse_input
# ---------------------------------------------------------------------------

class TestParseInput:
    def test_json_valido(self):
        result = _parse_input('{"periodo_inicio": "2026-04-01", "filtro_casa": "TBI"}')
        assert result["periodo_inicio"] == "2026-04-01"
        assert result["filtro_casa"] == "TBI"

    def test_kwargs_python(self):
        result = _parse_input("periodo_inicio='2026-04-01', filtro_casa='TBI'")
        assert result.get("periodo_inicio") == "2026-04-01"

    def test_json_invalido_fallback_regex(self):
        result = _parse_input("periodo_inicio=2026-04-01 filtro_casa=TBI")
        assert "periodo_inicio" in result


# ---------------------------------------------------------------------------
# _build_where
# ---------------------------------------------------------------------------

class TestBuildWhere:
    def test_sem_filtro(self):
        w = _build_where("2026-04-01", "2026-04-30", "", "", "")
        assert "BETWEEN '2026-04-01' AND '2026-04-30'" in w
        assert "unidade" not in w

    def test_filtro_casa(self):
        w = _build_where("2026-04-01", "2026-04-30", "TBI", "", "")
        assert "unidade = 'TBI'" in w

    def test_filtro_alavanca(self):
        w = _build_where("2026-04-01", "2026-04-30", "", "Tipo_A", "")
        assert "segmento = 'Tipo_A'" in w

    def test_filtro_marca(self):
        w = _build_where("2026-04-01", "2026-04-30", "", "", "MarcaX")
        assert "marca = 'MarcaX'" in w

    def test_date_col_personalizado(self):
        w = _build_where("2026-04-01", "2026-04-30", "", "", "", date_col="data")
        assert w.startswith("data BETWEEN")


# ---------------------------------------------------------------------------
# DremioResumoTool._run — validações de entrada
# ---------------------------------------------------------------------------

class TestDremioResumoToolValidacao:
    def _tool(self):
        return DremioResumoTool()

    def test_sem_periodo_retorna_erro(self):
        result = self._tool()._run("{}")
        assert "obrigatorios" in result.lower()

    def test_data_inicio_invalida(self):
        result = self._tool()._run('{"periodo_inicio": "2026-13-01", "periodo_fim": "2026-04-30"}')
        assert "invalida" in result.lower() or "nao e uma data" in result.lower()

    def test_data_fim_invalida(self):
        result = self._tool()._run('{"periodo_inicio": "2026-04-01", "periodo_fim": "abc"}')
        assert "invalida" in result.lower() or "nao e uma data" in result.lower()

    def test_periodo_invertido(self):
        result = self._tool()._run('{"periodo_inicio": "2026-04-30", "periodo_fim": "2026-04-01"}')
        assert "posterior" in result.lower() or "invalida" in result.lower() or "nao pode" in result.lower()

    def test_queries_executadas_em_paralelo(self):
        """Verifica que as 5 queries são submetidas ao pool com dados válidos."""
        mock_df_vendas = pd.DataFrame({
            "Grande_Grupo": ["ALIMENTOS"],
            "total_grupo": [1000.0],
            "fluxo_grupo": [50.0],
        })
        mock_df_vazio = pd.DataFrame()

        call_count = {"n": 0}

        def _mock_client(sql, max_wait=360):
            call_count["n"] += 1
            if "fTransacoes" in sql:
                return mock_df_vendas
            return mock_df_vazio

        with patch("src.tools.resumo_tool.client", side_effect=_mock_client):
            result = self._tool()._run(
                '{"periodo_inicio": "2026-04-01", "periodo_fim": "2026-04-30", "filtro_casa": "TBI"}'
            )

        assert call_count["n"] == 5
        assert "Resumo" in result
        assert "TBI" in result
