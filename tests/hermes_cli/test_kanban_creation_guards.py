"""Creation-time guards for structurally-dead Kanban cards."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_worktree_without_path_is_refused_at_creation(kanban_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="requires an explicit workspace_path") as exc:
            kb.create_task(conn, title="dead", workspace_kind="worktree")
        assert "unspawnable" in str(exc.value)
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_worktree_with_explicit_path_is_allowed(kanban_home, tmp_path):
    target = tmp_path / "repo"
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="valid worktree", workspace_kind="worktree",
            workspace_path=str(target), created_by="patch",
        )
        task = kb.get_task(conn, tid)
        assert task.workspace_kind == "worktree"
        assert task.workspace_path == str(target)
        assert task.created_by == "patch"


def test_owner_repo_placeholder_is_refused_with_value_and_example(kanban_home):
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="completion_contract 'OWNER/REPO' is a placeholder") as exc:
            kb.create_task(conn, title="placeholder", completion_contract="OWNER/REPO")
        assert "acme/hermes" in str(exc.value)
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_legitimate_completion_contracts_are_allowed(kanban_home):
    with kbc.connect() as conn:
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        repo = kb.create_task(conn, title="repo", completion_contract="acme/hermes")
        assert kb.get_task(conn, local).completion_contract == "local-only"
        assert kb.get_task(conn, repo).completion_contract == "acme/hermes"


def test_omitted_creator_is_explicitly_attributed_to_profile(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "patch")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="attributed")
        task = kb.get_task(conn, tid)
        assert task.created_by == "patch"
        event = [e for e in kb.list_events(conn, tid) if e.kind == "created"][-1]
        assert event.payload["created_by"] == "patch"
