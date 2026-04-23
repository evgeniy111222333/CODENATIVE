GRAPH_SHARED_NAME = "shared_graph_token"


def imported_helper(value: str) -> str:
    return value


class GraphWorker:
    def render(self, token: str) -> str:
        return f"graph::{token}"
