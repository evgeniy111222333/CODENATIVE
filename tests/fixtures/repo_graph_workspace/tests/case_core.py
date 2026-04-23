from app.core import build_payload


def test_build_payload_uses_graph_shared_name() -> None:
    payload = build_payload()
    assert payload["token"] == "shared_graph_token"
