from unittest.mock import patch, MagicMock

from src.app import _cmd_autorizar, _cmd_atualizar, _cmd_historico, _cmd_usuarios


def test_autorizar_usuario_valido():
    with patch("src.app.authorize", return_value="✅ ok") as mock_auth, \
         patch("src.app.get_user_nome", return_value="Admin"):
        _cmd_autorizar("5511999 ; João ; Dev ; SP", admin_phone="5511000")

    mock_auth.assert_called_once_with(
        phone="5511999",
        nome="João",
        cargo="Dev",
        casa="SP",
        added_by_tel="5511000",
        added_by_nome="Admin",
        admin=False,
    )


def test_autorizar_com_flag_admin():
    with patch("src.app.authorize", return_value="✅ ok") as mock_auth, \
         patch("src.app.get_user_nome", return_value="Admin"):
        _cmd_autorizar("5511999 ; João ; Dev ; SP ; admin", admin_phone="5511000")

    assert mock_auth.call_args.kwargs["admin"] is True


def test_autorizar_sem_flag_admin():
    with patch("src.app.authorize", return_value="✅ ok") as mock_auth, \
         patch("src.app.get_user_nome", return_value="Admin"):
        _cmd_autorizar("5511999 ; João ; Dev ; SP ; user", admin_phone="5511000")

    assert mock_auth.call_args.kwargs["admin"] is False


def test_autorizar_campos_insuficientes_retorna_aviso():
    result = _cmd_autorizar("5511999 ; João", admin_phone="5511000")
    assert result is not None and len(result) > 0


def test_autorizar_args_vazios_retorna_aviso():
    result = _cmd_autorizar("", admin_phone="5511000")
    assert result is not None and len(result) > 0


def test_autorizar_espaco_nos_campos_e_removido():
    with patch("src.app.authorize", return_value="✅ ok") as mock_auth, \
         patch("src.app.get_user_nome", return_value="Admin"):
        _cmd_autorizar("  5511999  ;  João  ;  Dev  ;  SP  ", admin_phone="5511000")

    assert mock_auth.call_args.kwargs["nome"] == "João"
    assert mock_auth.call_args.kwargs["phone"] == "5511999"


# ---------------------------------------------------------------------------
# _cmd_atualizar
# ---------------------------------------------------------------------------

def test_atualizar_telefone_valido():
    with patch("src.app.update_phone", return_value="✅ ok") as mock_upd:
        result = _cmd_atualizar("5511999 ; 5511888", admin_phone="5511000")
    mock_upd.assert_called_once_with("5511999", "5511888", updated_by="5511000")
    assert result == "✅ ok"


def test_atualizar_campos_insuficientes():
    result = _cmd_atualizar("5511999", admin_phone="5511000")
    assert result is not None and len(result) > 0


def test_atualizar_campo_vazio():
    result = _cmd_atualizar("", admin_phone="5511000")
    assert result is not None and len(result) > 0


# ---------------------------------------------------------------------------
# _cmd_historico
# ---------------------------------------------------------------------------

def test_historico_sem_dias_retorna_todo_historico():
    msgs = [{"role": "human", "content": "oi"}, {"role": "ai", "content": "olá"}]
    with patch("src.app.get_session_messages", return_value=msgs) as mock_hist, \
         patch("src.app.get_user_nome", return_value="João"):
        result = _cmd_historico("5511999")
    mock_hist.assert_called_once_with("5511999@s.whatsapp.net", since_ts=None)
    assert "João" in result
    assert "oi" in result


def test_historico_com_dias_filtra_por_timestamp():
    with patch("src.app.get_session_messages", return_value=[]) as mock_hist, \
         patch("src.app.get_user_nome", return_value="João"):
        result = _cmd_historico("5511999", days=7)
    args = mock_hist.call_args
    assert args[1]["since_ts"] is not None
    assert "Nenhum histórico" in result


def test_historico_vazio_retorna_mensagem():
    with patch("src.app.get_session_messages", return_value=[]), \
         patch("src.app.get_user_nome", return_value="5511999"):
        result = _cmd_historico("5511999")
    assert "Nenhum histórico" in result


# ---------------------------------------------------------------------------
# _cmd_usuarios
# ---------------------------------------------------------------------------

def test_usuarios_sem_cadastro():
    with patch("src.app.list_users", return_value=[]):
        result = _cmd_usuarios()
    assert "Nenhum" in result


def test_usuarios_lista_ativos():
    users = [
        {"telefone": "5511999", "nome": "Ana", "cargo": "Dev", "casa": "SP", "is_admin": 0, "active": 1},
    ]
    with patch("src.app.list_users", return_value=users):
        result = _cmd_usuarios(admin_only=False)
    assert "Ana" in result
    assert "5511999" in result


def test_usuarios_filtra_admin():
    users = [
        {"telefone": "5511999", "nome": "Ana", "cargo": "Dev", "casa": "SP", "is_admin": 0, "active": 1},
        {"telefone": "5511000", "nome": "Admin", "cargo": "TI", "casa": "RJ", "is_admin": 1, "active": 1},
    ]
    with patch("src.app.list_users", return_value=users):
        result = _cmd_usuarios(admin_only=True)
    assert "Admin" in result
    assert "Ana" not in result


def test_usuarios_lista_bloqueados():
    users = [
        {"telefone": "5511999", "nome": "João", "cargo": "Dev", "casa": "SP", "is_admin": 0, "active": 0},
    ]
    with patch("src.app.list_users", return_value=users):
        result = _cmd_usuarios(admin_only=False)
    assert "Bloqueados" in result
    assert "João" in result
