from app.helpers import GRAPH_SHARED_NAME, GraphWorker, imported_helper

SERVICE_ALIAS = "repo_graph_service"
PORT_VALUE = 4242


def build_payload() -> dict[str, object]:
    worker = GraphWorker()
    payload_token = GRAPH_SHARED_NAME
    worker.render(payload_token)
    imported_helper(payload_token)
    return {
        "service": SERVICE_ALIAS,
        "token": payload_token,
        "port": PORT_VALUE,
    }


def load_feature_flag() -> str:
    return "feature_toggle"
