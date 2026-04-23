import { uiToken, callUi } from "./util";

export const UI_MARKER = "widget_graph_marker";

export function renderWidget(): string {
  return callUi(uiToken + UI_MARKER);
}

test("widget uses graph token", () => {
  renderWidget();
});
