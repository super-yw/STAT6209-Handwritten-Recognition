from langgraph.graph import StateGraph, END, START
from .state import AgentState
from .edges import after_count, after_loc, after_num
from .nodes import pre_process_node, count_node, loc_node, get_number_node, revise_mod_node, revise_count_node, revise_loc_node, get_json_node

def build_graph():
    workflow = StateGraph(AgentState)
    # ── 添加节点 ──────────────────────────────
    workflow.add_node("pre_process", pre_process_node)
    workflow.add_node("count", count_node)
    workflow.add_node("loc", loc_node)
    workflow.add_node("get_number", get_number_node)
    workflow.add_node("revise_num", revise_mod_node)
    workflow.add_node("revise_count", revise_count_node)
    workflow.add_node("revise_loc", revise_loc_node)
    workflow.add_node("get_json", get_json_node)

    # ── 添加边 ────────────────────────────────
    workflow.add_edge(START,"pre_process")
    workflow.add_edge("pre_process", "count")
    workflow.add_edge("count", "revise_count")
    workflow.add_conditional_edges("revise_count", after_count, {
    "bad qly": "count",
    "no modif": "get_json",
    "modif exist": "loc"
    })
    workflow.add_edge("loc", "revise_loc")
    workflow.add_conditional_edges("revise_loc", after_loc, {
    "bad qly": "loc",
    "good qly": "get_number"
    })
    workflow.add_edge("get_number", "revise_num")
    workflow.add_conditional_edges("revise_num", after_num, {
    "bad qly": "get_number",
    "good qly": "get_json"
    })
    workflow.add_edge("get_json", END)

    return workflow.compile()


# 编译好的 graph 实例
agent = build_graph()