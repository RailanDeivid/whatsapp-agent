import pytest

from src import access_control


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Banco SQLite isolado em diretório temporário para cada teste."""
    db_path = str(tmp_path / "test_access.db")
    monkeypatch.setattr(access_control, "SQLITE_PATH", db_path)
    monkeypatch.setattr("src.config.SEED_USERS", [])
    access_control.init_db()
    return db_path


def test_usuario_nao_cadastrado_nao_autorizado(db):
    assert not access_control.is_authorized("5511999")


def test_autorizar_usuario(db):
    result = access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    assert "autorizado" in result
    assert access_control.is_authorized("5511999")


def test_revogar_usuario(db):
    access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    result = access_control.revoke("5511999", "5511000")
    assert "bloqueado" in result
    assert not access_control.is_authorized("5511999")


def test_desbloquear_usuario(db):
    access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    access_control.revoke("5511999", "5511000")
    result = access_control.unblock("5511999", "5511000")
    assert "desbloqueado" in result
    assert access_control.is_authorized("5511999")


def test_remover_usuario(db):
    access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    result = access_control.delete_user("5511999", "5511000")
    assert "removido" in result
    assert not access_control.is_authorized("5511999")


def test_is_admin_verdadeiro(db):
    access_control.authorize("5511999", "Admin", "TI", "SP", "5511000", admin=True)
    assert access_control.is_admin("5511999")


def test_is_admin_falso(db):
    access_control.authorize("5511999", "User", "Dev", "SP", "5511000", admin=False)
    assert not access_control.is_admin("5511999")


def test_admin_bloqueado_nao_e_admin(db):
    access_control.authorize("5511999", "Admin", "TI", "SP", "5511000", admin=True)
    access_control.revoke("5511999", "5511000")
    assert not access_control.is_admin("5511999")


def test_revogar_inexistente_retorna_aviso(db):
    result = access_control.revoke("5599999", "5511000")
    assert "⚠️" in result


def test_revogar_ja_bloqueado_retorna_aviso(db):
    access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    access_control.revoke("5511999", "5511000")
    result = access_control.revoke("5511999", "5511000")
    assert "⚠️" in result


def test_reautorizar_usuario_bloqueado(db):
    access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    access_control.revoke("5511999", "5511000")
    result = access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    assert "reativado" in result
    assert access_control.is_authorized("5511999")


def test_get_user_nome(db):
    access_control.authorize("5511999", "Maria", "RH", "RJ", "5511000")
    assert access_control.get_user_nome("5511999") == "Maria"


def test_get_user_nome_inexistente_retorna_telefone(db):
    assert access_control.get_user_nome("5599000") == "5599000"


def test_update_phone_valido(db):
    access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    result = access_control.update_phone("5511999", "5511888", "5511000")
    assert "5511888" in result
    assert access_control.is_authorized("5511888")
    assert not access_control.is_authorized("5511999")


def test_update_phone_inexistente(db):
    result = access_control.update_phone("5599999", "5511888", "5511000")
    assert "⚠️" in result


def test_update_phone_destino_ja_existe(db):
    access_control.authorize("5511999", "João", "Dev", "SP", "5511000")
    access_control.authorize("5511888", "Maria", "RH", "RJ", "5511000")
    result = access_control.update_phone("5511999", "5511888", "5511000")
    assert "⚠️" in result


def test_list_users_retorna_todos(db):
    access_control.authorize("5511111", "Ana", "Dev", "SP", "5511000")
    access_control.authorize("5511222", "Bob", "TI", "RJ", "5511000", admin=True)
    users = access_control.list_users()
    telefones = [u["telefone"] for u in users]
    assert "5511111" in telefones
    assert "5511222" in telefones


def test_list_users_vazio_sem_cadastro(db):
    assert access_control.list_users() == []
