from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_plugin_slash_handler_receives_real_gateway_session_context():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._hm_quick_commands = lambda: {}

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        user_id="user-1",
        user_name="Founder",
    )
    event = SimpleNamespace(get_command_args=lambda: "reason")
    observed = {}

    def _handler(raw_args):
        from gateway.session_context import get_session_env
        from tools.approval_context import get_current_session_key

        observed.update(
            args=raw_args,
            platform=get_session_env("HERMES_SESSION_PLATFORM", ""),
            user_id=get_session_env("HERMES_SESSION_USER_ID", ""),
            session_key=get_session_env("HERMES_SESSION_KEY", ""),
            approval_key=get_current_session_key("missing"),
        )
        return "ok"

    with patch("hermes_cli.plugins.get_plugin_command_handler", return_value=_handler):
        handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
            event, source, "sona-direct", "telegram-session-1"
        )

    assert (handled, result, command) == (True, "ok", "sona-direct")
    assert observed == {
        "args": "reason",
        "platform": "telegram",
        "user_id": "user-1",
        "session_key": "telegram-session-1",
        "approval_key": "telegram-session-1",
    }
