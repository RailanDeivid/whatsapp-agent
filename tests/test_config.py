from src.config import _parse_seed_users


def test_usuario_simples():
    result = _parse_seed_users("5511999:João:Dev:SP")
    assert result == [{"telefone": "5511999", "nome": "João", "cargo": "Dev", "casa": "SP", "is_admin": 0}]


def test_usuario_admin():
    result = _parse_seed_users("5511999:João:Dev:SP:admin")
    assert result[0]["is_admin"] == 1


def test_usuario_nao_admin_explicito():
    result = _parse_seed_users("5511999:João:Dev:SP:user")
    assert result[0]["is_admin"] == 0


def test_multiplos_usuarios():
    raw = "5511111:Ana:RH:RJ, 5511222:Bob:TI:SP:admin"
    result = _parse_seed_users(raw)
    assert len(result) == 2
    assert result[0]["telefone"] == "5511111"
    assert result[1]["is_admin"] == 1


def test_entrada_vazia():
    assert _parse_seed_users("") == []


def test_entrada_com_espacos_extras():
    result = _parse_seed_users("  5511999 : João : Dev : SP  ")
    assert result[0]["telefone"] == "5511999"
    assert result[0]["nome"] == "João"


def test_entrada_incompleta_ignorada():
    # Menos de 4 campos — deve ser ignorada
    assert _parse_seed_users("5511999:João:Dev") == []


def test_mistura_valido_e_invalido():
    raw = "5511111:Ana:RH:RJ, invalido, 5511222:Bob:TI:SP"
    result = _parse_seed_users(raw)
    assert len(result) == 2
