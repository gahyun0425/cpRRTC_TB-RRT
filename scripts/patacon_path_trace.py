#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
ET.register_namespace("", GRAPHML_NS)


GRAPH_KEYS = {
    "planner": "g_planner",
    "trace_level": "g_trace_level",
    "dimension": "g_dimension",
    "max_grow_step": "g_max_grow_step",
    "max_display_step": "g_max_display_step",
    "max_parallel_step": "g_max_parallel_step",
    "joint_names": "g_joint_names",
    "solution_order": "g_solution_order",
    "slot_steps_json": "g_slot_steps_json",
    "timeline_events_json": "g_timeline_events_json",
}

NODE_KEYS = {
    "seq": "n_seq",
    "tree": "n_tree",
    "batch_idx": "n_batch_idx",
    "node_idx": "n_node_idx",
    "parent_idx": "n_parent_idx",
    "parent_id": "n_parent_id",
    "iter": "n_iter",
    "phase": "n_phase",
    "step_type": "n_step_type",
    "slot_idx": "n_slot_idx",
    "escape_step": "n_escape_step",
    "ts_id": "n_ts_id",
    "is_proj_root": "n_is_proj_root",
    "grow_step": "n_grow_step",
    "duration_sec": "n_duration_sec",
    "display_step": "n_display_step",
    "parallel_step": "n_parallel_step",
    "parallel_step_start_sec": "n_parallel_step_start_sec",
    "parallel_step_finished_at_sec": "n_parallel_step_finished_at_sec",
    "parallel_step_duration_sec": "n_parallel_step_duration_sec",
    "depth": "n_depth",
    "solution": "n_solution",
    "event_kind": "n_event_kind",
    "active": "n_active",
    "advanced": "n_advanced",
    "trapped": "n_trapped",
    "reached": "n_reached",
    "mean_progress": "n_mean_progress",
    "max_progress": "n_max_progress",
    "simultaneous": "n_simultaneous",
    "simultaneous_group": "n_simultaneous_group",
    "order_index": "n_order_index",
    "q_json": "n_q_json",
}

EDGE_KEYS = {
    "kind": "e_kind",
    "tree": "e_tree",
    "batch_idx": "e_batch_idx",
    "iter": "e_iter",
    "phase": "e_phase",
    "grow_step": "e_grow_step",
    "display_step": "e_display_step",
    "parallel_step": "e_parallel_step",
    "solution": "e_solution",
}


def tag(name: str) -> str:
    return f"{{{GRAPHML_NS}}}{name}"


def data(parent: ET.Element, key: str, value: Any) -> ET.Element:
    elem = ET.SubElement(parent, tag("data"), {"key": key})
    if isinstance(value, bool):
        elem.text = "true" if value else "false"
    elif isinstance(value, (dict, list)):
        elem.text = json.dumps(value)
    else:
        elem.text = str(value)
    return elem


def load_result(path: Path, result_index: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" in payload:
        results = payload["results"]
        if not isinstance(results, list) or not results:
            raise ValueError(f"{path} does not contain any saved results")
        if result_index < 0 or result_index >= len(results):
            raise IndexError(f"result index {result_index} is outside 0..{len(results) - 1}")
        return results[result_index]
    return payload


def normalized_path(result: dict[str, Any], path_key: str) -> list[list[float]]:
    raw_path = result.get(path_key)
    if raw_path is None and path_key != "path":
        raw_path = result.get("path")
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        raise ValueError(f"result does not contain a usable path under {path_key!r}")

    out: list[list[float]] = []
    for idx, row in enumerate(raw_path):
        if not isinstance(row, list):
            raise ValueError(f"path waypoint {idx} is not a list")
        out.append([float(value) for value in row])
    return out


def add_graphml_keys(root: ET.Element, dimension: int) -> None:
    keys = [
        ("graph", GRAPH_KEYS["planner"], "planner", "string"),
        ("graph", GRAPH_KEYS["trace_level"], "trace_level", "string"),
        ("graph", GRAPH_KEYS["dimension"], "dimension", "int"),
        ("graph", GRAPH_KEYS["max_grow_step"], "max_grow_step", "int"),
        ("graph", GRAPH_KEYS["max_display_step"], "max_display_step", "int"),
        ("graph", GRAPH_KEYS["max_parallel_step"], "max_parallel_step", "int"),
        ("graph", GRAPH_KEYS["joint_names"], "joint_names", "string"),
        ("graph", GRAPH_KEYS["solution_order"], "solution_order", "string"),
        ("graph", GRAPH_KEYS["slot_steps_json"], "slot_steps_json", "string"),
        ("graph", GRAPH_KEYS["timeline_events_json"], "timeline_events_json", "string"),
        ("node", NODE_KEYS["seq"], "seq", "int"),
        ("node", NODE_KEYS["tree"], "tree", "string"),
        ("node", NODE_KEYS["batch_idx"], "batch_idx", "int"),
        ("node", NODE_KEYS["node_idx"], "node_idx", "int"),
        ("node", NODE_KEYS["parent_idx"], "parent_idx", "int"),
        ("node", NODE_KEYS["parent_id"], "parent_id", "string"),
        ("node", NODE_KEYS["iter"], "iter", "int"),
        ("node", NODE_KEYS["phase"], "phase", "string"),
        ("node", NODE_KEYS["step_type"], "step_type", "string"),
        ("node", NODE_KEYS["slot_idx"], "slot_idx", "int"),
        ("node", NODE_KEYS["escape_step"], "escape_step", "int"),
        ("node", NODE_KEYS["ts_id"], "ts_id", "int"),
        ("node", NODE_KEYS["is_proj_root"], "is_proj_root", "boolean"),
        ("node", NODE_KEYS["grow_step"], "grow_step", "int"),
        ("node", NODE_KEYS["duration_sec"], "duration_sec", "double"),
        ("node", NODE_KEYS["display_step"], "display_step", "int"),
        ("node", NODE_KEYS["parallel_step"], "parallel_step", "int"),
        ("node", NODE_KEYS["parallel_step_start_sec"], "parallel_step_start_sec", "double"),
        ("node", NODE_KEYS["parallel_step_finished_at_sec"], "parallel_step_finished_at_sec", "double"),
        ("node", NODE_KEYS["parallel_step_duration_sec"], "parallel_step_duration_sec", "double"),
        ("node", NODE_KEYS["depth"], "depth", "int"),
        ("node", NODE_KEYS["solution"], "solution", "boolean"),
        ("node", NODE_KEYS["event_kind"], "event_kind", "string"),
        ("node", NODE_KEYS["active"], "active", "int"),
        ("node", NODE_KEYS["advanced"], "advanced", "int"),
        ("node", NODE_KEYS["trapped"], "trapped", "int"),
        ("node", NODE_KEYS["reached"], "reached", "int"),
        ("node", NODE_KEYS["mean_progress"], "mean_progress", "double"),
        ("node", NODE_KEYS["max_progress"], "max_progress", "double"),
        ("node", NODE_KEYS["simultaneous"], "simultaneous", "boolean"),
        ("node", NODE_KEYS["simultaneous_group"], "simultaneous_group", "string"),
        ("node", NODE_KEYS["order_index"], "order_index", "int"),
        ("node", NODE_KEYS["q_json"], "q_json", "string"),
        ("edge", EDGE_KEYS["kind"], "kind", "string"),
        ("edge", EDGE_KEYS["tree"], "tree", "string"),
        ("edge", EDGE_KEYS["batch_idx"], "batch_idx", "int"),
        ("edge", EDGE_KEYS["iter"], "iter", "int"),
        ("edge", EDGE_KEYS["phase"], "phase", "string"),
        ("edge", EDGE_KEYS["grow_step"], "grow_step", "int"),
        ("edge", EDGE_KEYS["display_step"], "display_step", "int"),
        ("edge", EDGE_KEYS["parallel_step"], "parallel_step", "int"),
        ("edge", EDGE_KEYS["solution"], "solution", "boolean"),
    ]
    for idx in range(dimension):
        keys.append(("node", f"n_q{idx}", f"q{idx}", "double"))
    for scope, key_id, attr_name, attr_type in keys:
        ET.SubElement(
            root,
            tag("key"),
            {
                "id": key_id,
                "for": scope,
                "attr.name": attr_name,
                "attr.type": attr_type,
            },
        )


def tree_node_id(tree: str, idx: int) -> str:
    return f"n_{tree}_i{idx}"


def has_tree_trace(result: dict[str, Any]) -> bool:
    trace = result.get("tree_trace")
    return isinstance(trace, dict) and isinstance(trace.get("trees"), list)


def solution_history_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("solution_history")
    if not isinstance(rows, list):
        return []
    output = [row for row in rows if isinstance(row, dict)]
    output.sort(key=lambda row: int(row.get("update_index", 0)))
    return output


def solution_node_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_order = row.get("solution_order")
    if not isinstance(raw_order, list):
        return []
    output = [item for item in raw_order if isinstance(item, dict)]
    output.sort(key=lambda item: int(item.get("order", 0)))
    return output


def tree_depth(nodes_by_idx: dict[int, dict[str, Any]], idx: int) -> int:
    depth = 0
    current = idx
    seen: set[int] = set()
    while current in nodes_by_idx and current not in seen:
        seen.add(current)
        parent = int(nodes_by_idx[current].get("parent_idx", -1))
        if parent == current or parent < 0:
            break
        current = parent
        depth += 1
    return depth


def compact_tree_nodes(
    nodes_by_tree: dict[str, dict[int, dict[str, Any]]],
    protected_node_ids: set[str],
    max_tree_nodes: int | None,
) -> dict[str, dict[int, dict[str, Any]]]:
    if max_tree_nodes is None or max_tree_nodes <= 0:
        return nodes_by_tree

    total_nodes = sum(len(nodes) for nodes in nodes_by_tree.values())
    if total_nodes <= max_tree_nodes:
        return nodes_by_tree

    selected_by_tree: dict[str, set[int]] = {tree: set() for tree in nodes_by_tree}
    for node_id in protected_node_ids:
        for tree_name, nodes in nodes_by_tree.items():
            prefix = f"n_{tree_name}_i"
            if node_id.startswith(prefix):
                idx = int(node_id[len(prefix):])
                if idx in nodes:
                    selected_by_tree[tree_name].add(idx)
                break

    remaining = max(0, max_tree_nodes - sum(len(nodes) for nodes in selected_by_tree.values()))
    non_solution_total = sum(
        max(0, len(nodes_by_tree[tree]) - len(selected_by_tree[tree]))
        for tree in nodes_by_tree
    )
    for tree_name, nodes in nodes_by_tree.items():
        if not nodes:
            continue
        selected = selected_by_tree[tree_name]
        if 0 in nodes:
            selected.add(0)
        non_solution = max(0, len(nodes) - len(selected))
        allowance = int(round(remaining * non_solution / max(1, non_solution_total)))
        allowance = max(0, min(allowance, non_solution))
        candidates = [idx for idx in sorted(nodes) if idx not in selected]
        if allowance >= len(candidates):
            selected.update(candidates)
        elif allowance > 0 and candidates:
            step = max(1, len(candidates) // allowance)
            selected.update(candidates[::step][:allowance])

    return {
        tree_name: {idx: nodes[idx] for idx in sorted(selected) if idx in nodes}
        for tree_name, nodes in nodes_by_tree.items()
    }


def tree_trace_graphml_text(
    result: dict[str, Any],
    *,
    max_tree_nodes: int | None = None,
) -> str:
    trace = result.get("tree_trace")
    if not isinstance(trace, dict):
        raise ValueError("result does not contain tree_trace")
    trees = trace.get("trees")
    if not isinstance(trees, list):
        raise ValueError("tree_trace does not contain a trees list")

    dimension = int(result.get("dimension") or 0)
    joint_names = result.get("joint_names")
    if not isinstance(joint_names, list) or (dimension and len(joint_names) != dimension):
        joint_names = [f"q{i}" for i in range(dimension)]
    if dimension == 0 and trees:
        for tree in trees:
            nodes = tree.get("nodes", [])
            if nodes:
                dimension = len(nodes[0].get("q", []))
                joint_names = [f"q{i}" for i in range(dimension)]
                break

    solution_order_rows = trace.get("solution_order", [])
    solution_ids: list[str] = []
    solution_order_by_id: dict[str, int] = {}
    if isinstance(solution_order_rows, list):
        ordered_rows = sorted(
            [row for row in solution_order_rows if isinstance(row, dict)],
            key=lambda row: int(row.get("order", 0)),
        )
        for order, row in enumerate(ordered_rows):
            tree = str(row.get("tree") or ("start" if int(row.get("tree_id", 0)) == 0 else "goal"))
            idx = int(row.get("idx"))
            node_id = tree_node_id(tree, idx)
            solution_ids.append(node_id)
            solution_order_by_id[node_id] = order
    solution_pair_keys = {
        frozenset((solution_ids[i], solution_ids[i + 1]))
        for i in range(max(0, len(solution_ids) - 1))
    }
    # AORRTC restarts its trees after every improved solution.  Therefore
    # solution_history[*].solution_order contains node indices that belong to
    # *old* trees and must not be matched against the final/best tree by index.
    # Keep previous solutions using their actual configuration trajectories.
    history_paths: list[dict[str, Any]] = []
    history_rows = solution_history_rows(result)
    for history_index, row in enumerate(history_rows):
        raw_path = row.get("path_start_to_goal")
        if not isinstance(raw_path, list) or len(raw_path) < 2:
            continue

        path: list[list[float]] = []
        valid_path = True
        for waypoint in raw_path:
            if not isinstance(waypoint, list):
                valid_path = False
                break
            q = [float(value) for value in waypoint]
            if dimension > 0 and len(q) != dimension:
                valid_path = False
                break
            path.append(q)

        if not valid_path or len(path) < 2:
            continue

        history_paths.append({
            "update_index": int(row.get("update_index", history_index)),
            "iteration": int(row.get("iteration", row.get("iter", 0))),
            "cost": float(row.get("cost", 0.0)),
            # The last retained update is the best/final solution and is
            # already rendered using the actual final tree solution chain.
            "final": history_index == len(history_rows) - 1,
            "path": path,
        })

    root = ET.Element(tag("graphml"))
    add_graphml_keys(root, dimension)
    graph = ET.SubElement(root, tag("graph"), {"id": "patacon_trace", "edgedefault": "directed"})

    ready_nodes_by_tree: dict[str, dict[int, dict[str, Any]]] = {}
    max_step = 0
    total_nodes = 0
    for tree in trees:
        tree_name = str(tree.get("name", "start"))
        nodes = {
            int(node.get("idx")): node
            for node in tree.get("nodes", [])
            if isinstance(node, dict) and bool(node.get("ready", True))
        }
        ready_nodes_by_tree[tree_name] = nodes
        max_step = max(max_step, max(nodes.keys(), default=0))
        total_nodes += len(nodes)
    # Tree compaction protects only the final/best solution nodes.
    # Previous AORRTC solutions are rendered as independent path overlays.
    ready_nodes_by_tree = compact_tree_nodes(
        ready_nodes_by_tree,
        set(solution_ids),
        max_tree_nodes,
    )

    data(graph, GRAPH_KEYS["planner"], str(result.get("planner", "PATACON")))
    data(graph, GRAPH_KEYS["trace_level"], "nodes")
    data(graph, GRAPH_KEYS["dimension"], dimension)
    data(graph, GRAPH_KEYS["max_grow_step"], max_step)
    data(graph, GRAPH_KEYS["max_display_step"], max_step)
    data(
        graph,
        GRAPH_KEYS["max_parallel_step"],
        max(
            max_step,
            max(
                (int(path["update_index"]) for path in history_paths),
                default=0,
            ),
        ),
    )
    data(graph, GRAPH_KEYS["joint_names"], json.dumps(joint_names))
    data(graph, GRAPH_KEYS["solution_order"], json.dumps(solution_ids))
    data(graph, GRAPH_KEYS["slot_steps_json"], "[]")
    data(graph, GRAPH_KEYS["timeline_events_json"], "[]")

    seq = 0
    for tree in trees:
        tree_id = int(tree.get("tree_id", 0))
        tree_name = str(tree.get("name", "start"))
        nodes = ready_nodes_by_tree.get(tree_name, {})
        for idx in sorted(nodes):
            node_payload = nodes[idx]
            q = [float(value) for value in node_payload.get("q", [])]
            parent_idx = int(node_payload.get("parent_idx", -1))
            parent_id = (
                tree_node_id(tree_name, parent_idx)
                if parent_idx >= 0 and parent_idx != idx and parent_idx in nodes
                else ""
            )
            node_id = tree_node_id(tree_name, idx)
            order_index = solution_order_by_id.get(node_id, -1)
            # Only the final/best search tree solution is marked as the
            # tree solution.  Old AORRTC paths are separate overlays.
            is_solution_node = order_index >= 0
            elem = ET.SubElement(graph, tag("node"), {"id": node_id})
            node_values = {
                "seq": seq,
                "tree": tree_name,
                "batch_idx": tree_id,
                "node_idx": idx,
                "parent_idx": parent_idx,
                "parent_id": parent_id,
                "iter": idx,
                "phase": "tree_growth",
                "step_type": "connect" if is_solution_node else "extend",
                "slot_idx": 0,
                "escape_step": -1,
                "ts_id": -1,
                "is_proj_root": parent_idx == idx,
                "grow_step": idx,
                "duration_sec": 0.0,
                "display_step": idx,
                "parallel_step": idx,
                "parallel_step_start_sec": 0.0,
                "parallel_step_finished_at_sec": 0.0,
                "parallel_step_duration_sec": 0.0,
                "depth": tree_depth(nodes, idx),
                "solution": is_solution_node,
                "event_kind": "node_add",
                "active": 1,
                "advanced": 1 if parent_id else 0,
                "trapped": 0,
                "reached": order_index == len(solution_ids) - 1 and order_index >= 0,
                "mean_progress": idx / max(1, max_step),
                "max_progress": idx / max(1, max_step),
                "simultaneous": False,
                "simultaneous_group": "",
                "order_index": order_index,
            }
            for key, value in node_values.items():
                data(elem, NODE_KEYS[key], value)
            data(elem, NODE_KEYS["q_json"], json.dumps(q))
            for q_idx, value in enumerate(q):
                data(elem, f"n_q{q_idx}", value)
            seq += 1

    edge_idx = 0
    for tree in trees:
        tree_id = int(tree.get("tree_id", 0))
        tree_name = str(tree.get("name", "start"))
        nodes = ready_nodes_by_tree.get(tree_name, {})
        for idx in sorted(nodes):
            parent_idx = int(nodes[idx].get("parent_idx", -1))
            if parent_idx < 0 or parent_idx == idx or parent_idx not in nodes:
                continue
            source = tree_node_id(tree_name, parent_idx)
            target = tree_node_id(tree_name, idx)
            solution = frozenset((source, target)) in solution_pair_keys
            elem = ET.SubElement(
                graph,
                tag("edge"),
                {"id": f"e_tree_{edge_idx}", "source": source, "target": target},
            )
            edge_values = {
                "kind": "tree",
                "tree": tree_name,
                "batch_idx": tree_id,
                "iter": idx,
                "phase": "tree_growth",
                "grow_step": idx,
                "display_step": idx,
                "parallel_step": idx,
                "solution": solution,
            }
            for key, value in edge_values.items():
                data(elem, EDGE_KEYS[key], value)
            edge_idx += 1

    connection = trace.get("connection")
    if isinstance(connection, dict):
        source_tree = str(connection.get("source_tree", ""))
        target_tree = str(connection.get("target_tree", ""))
        source_idx = int(connection.get("source_idx", -1))
        target_idx = int(connection.get("target_idx", -1))
        if source_tree == "goal":
            source_tree, target_tree = target_tree, source_tree
            source_idx, target_idx = target_idx, source_idx
        if source_tree and target_tree and source_idx >= 0 and target_idx >= 0:
            source = tree_node_id(source_tree, source_idx)
            target = tree_node_id(target_tree, target_idx)
            elem = ET.SubElement(
                graph,
                tag("edge"),
                {"id": f"e_connection_{edge_idx}", "source": source, "target": target},
            )
            step = max(source_idx, target_idx)
            edge_values = {
                "kind": "connection",
                "tree": "connection",
                "batch_idx": -1,
                "iter": step,
                "phase": "connection",
                "grow_step": step,
                "display_step": step,
                "parallel_step": step,
                "solution": True,
            }
            for key, value in edge_values.items():
                data(elem, EDGE_KEYS[key], value)
            edge_idx += 1

    # Add previous AORRTC solutions as synthetic configuration-only path
    # overlays.  Their waypoint nodes are intentionally not tree nodes and
    # will be hidden by the HTML viewer; only the colored path edges remain.
    # The final/best solution is not duplicated because it already exists in
    # the final tree as the regular green solution chain.
    for history_path in history_paths:
        if bool(history_path["final"]):
            continue

        update_index = int(history_path["update_index"])
        iteration = int(history_path["iteration"])
        cost = float(history_path["cost"])
        path = history_path["path"]
        path_node_ids: list[str] = []

        for state_index, q in enumerate(path):
            node_id = f"n_aorrtc_history_u{update_index}_s{state_index}"
            path_node_ids.append(node_id)
            parent_id = "" if state_index == 0 else path_node_ids[state_index - 1]

            elem = ET.SubElement(graph, tag("node"), {"id": node_id})
            progress = state_index / max(1, len(path) - 1)
            node_values = {
                "seq": seq,
                "tree": "history",
                "batch_idx": update_index,
                "node_idx": state_index,
                "parent_idx": -1 if state_index == 0 else state_index - 1,
                "parent_id": parent_id,
                "iter": iteration,
                "phase": "aorrtc_history_path",
                "step_type": "connect",
                "slot_idx": update_index,
                "escape_step": -1,
                "ts_id": -1,
                "is_proj_root": state_index == 0,
                "grow_step": max_step,
                "duration_sec": 0.0,
                # Show history overlays only at the final display step so the
                # best-tree growth animation remains unchanged.
                "display_step": max_step,
                "parallel_step": update_index,
                "parallel_step_start_sec": 0.0,
                "parallel_step_finished_at_sec": 0.0,
                "parallel_step_duration_sec": 0.0,
                "depth": state_index,
                "solution": False,
                "event_kind": "aorrtc_history_path",
                "active": 1,
                "advanced": 1 if state_index > 0 else 0,
                "trapped": 0,
                "reached": 1 if state_index == len(path) - 1 else 0,
                "mean_progress": progress,
                "max_progress": progress,
                "simultaneous": False,
                "simultaneous_group": (
                    f"aorrtc_history_u{update_index}_cost_{cost:.6f}"
                ),
                "order_index": -1,
            }
            for key, value in node_values.items():
                data(elem, NODE_KEYS[key], value)
            data(elem, NODE_KEYS["q_json"], json.dumps(q))
            for q_idx, value in enumerate(q):
                data(elem, f"n_q{q_idx}", value)
            seq += 1

        for state_index in range(1, len(path)):
            source = path_node_ids[state_index - 1]
            target = path_node_ids[state_index]
            elem = ET.SubElement(
                graph,
                tag("edge"),
                {
                    "id": (
                        f"e_aorrtc_history_u{update_index}_"
                        f"s{state_index - 1}_{edge_idx}"
                    ),
                    "source": source,
                    "target": target,
                },
            )
            edge_values = {
                "kind": "solution_update",
                "tree": "history",
                "batch_idx": update_index,
                "iter": iteration,
                "phase": "aorrtc_solution_update",
                "grow_step": max_step,
                "display_step": max_step,
                "parallel_step": update_index,
                "solution": True,
            }
            for key, value in edge_values.items():
                data(elem, EDGE_KEYS[key], value)
            edge_idx += 1

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def graphml_text(result: dict[str, Any], path: list[list[float]]) -> str:
    dimension = int(result.get("dimension") or len(path[0]))
    if any(len(row) != dimension for row in path):
        raise ValueError("not all path waypoints match the result dimension")

    joint_names = result.get("joint_names")
    if not isinstance(joint_names, list) or len(joint_names) != dimension:
        joint_names = [f"q{i}" for i in range(dimension)]

    root = ET.Element(tag("graphml"))
    add_graphml_keys(root, dimension)
    graph = ET.SubElement(root, tag("graph"), {"id": "patacon_trace", "edgedefault": "directed"})
    max_step = len(path) - 1
    solution_order = [f"n_path_{idx}" for idx in range(len(path))]
    slot_steps = [
        {
            "seq": idx,
            "iter": idx,
            "tree": "start",
            "batch_idx": 0,
            "slot_idx": 0,
            "phase": "solution_path",
            "step": "path_edge",
            "step_type": "connect",
            "result": "advanced",
            "substep": 0,
            "grow_step": idx,
            "display_step": idx,
            "parallel_step": idx,
            "duration_sec": 0.0,
            "progress": idx / max(1, max_step),
        }
        for idx in range(1, len(path))
    ]

    data(graph, GRAPH_KEYS["planner"], str(result.get("planner", "PATACON")))
    data(graph, GRAPH_KEYS["trace_level"], "nodes")
    data(graph, GRAPH_KEYS["dimension"], dimension)
    data(graph, GRAPH_KEYS["max_grow_step"], max_step)
    data(graph, GRAPH_KEYS["max_display_step"], max_step)
    data(graph, GRAPH_KEYS["max_parallel_step"], max_step)
    data(graph, GRAPH_KEYS["joint_names"], json.dumps(joint_names))
    data(graph, GRAPH_KEYS["solution_order"], json.dumps(solution_order))
    data(graph, GRAPH_KEYS["slot_steps_json"], json.dumps(slot_steps))
    data(graph, GRAPH_KEYS["timeline_events_json"], "[]")

    for idx, q in enumerate(path):
        node = ET.SubElement(graph, tag("node"), {"id": f"n_path_{idx}"})
        node_values = {
            "seq": idx,
            "tree": "start",
            "batch_idx": 0,
            "node_idx": idx,
            "parent_idx": idx - 1,
            "parent_id": "" if idx == 0 else f"n_path_{idx - 1}",
            "iter": idx,
            "phase": "solution_path",
            "step_type": "connect",
            "slot_idx": 0,
            "escape_step": -1,
            "ts_id": -1,
            "is_proj_root": idx == 0,
            "grow_step": idx,
            "duration_sec": 0.0,
            "display_step": idx,
            "parallel_step": idx,
            "parallel_step_start_sec": 0.0,
            "parallel_step_finished_at_sec": 0.0,
            "parallel_step_duration_sec": 0.0,
            "depth": idx,
            "solution": True,
            "event_kind": "node_add",
            "active": 1,
            "advanced": 1 if idx > 0 else 0,
            "trapped": 0,
            "reached": 1 if idx == len(path) - 1 else 0,
            "mean_progress": idx / max(1, max_step),
            "max_progress": idx / max(1, max_step),
            "simultaneous": False,
            "simultaneous_group": "",
            "order_index": idx,
        }
        for key, value in node_values.items():
            data(node, NODE_KEYS[key], value)
        data(node, NODE_KEYS["q_json"], json.dumps(q))
        for q_idx, value in enumerate(q):
            data(node, f"n_q{q_idx}", value)

    for idx in range(1, len(path)):
        edge = ET.SubElement(
            graph,
            tag("edge"),
            {
                "id": f"e_path_{idx - 1}",
                "source": f"n_path_{idx - 1}",
                "target": f"n_path_{idx}",
            },
        )
        edge_values = {
            "kind": "tree",
            "tree": "start",
            "batch_idx": 0,
            "iter": idx,
            "phase": "solution_path",
            "grow_step": idx,
            "display_step": idx,
            "parallel_step": idx,
            "solution": True,
        }
        for key, value in edge_values.items():
            data(edge, EDGE_KEYS[key], value)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def generated_paths_graphml_text(
    result: dict[str, Any],
    *,
    max_paths: int | None = None,
) -> tuple[str, int, int]:
    raw_paths = result.get("solution_history")
    if not isinstance(raw_paths, list) or not raw_paths:
        raw_paths = result.get("generated_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(
            "result does not contain solution_history or generated_paths; "
            "rerun AORRTC with tracing enabled"
        )

    paths: list[tuple[dict[str, Any], list[list[float]]]] = []
    for row in raw_paths:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("path_start_to_goal")
        if not isinstance(raw_path, list) or len(raw_path) < 2:
            continue
        path = [[float(value) for value in waypoint] for waypoint in raw_path]
        paths.append((row, path))
        if max_paths is not None and max_paths > 0 and len(paths) >= max_paths:
            break

    if not paths:
        raise ValueError("generated_paths did not contain any path_start_to_goal entries")

    dimension = int(result.get("dimension") or len(paths[0][1][0]))
    for _, path in paths:
        if any(len(row) != dimension for row in path):
            raise ValueError("not all generated path waypoints match the result dimension")

    joint_names = result.get("joint_names")
    if not isinstance(joint_names, list) or len(joint_names) != dimension:
        joint_names = [f"q{i}" for i in range(dimension)]

    root = ET.Element(tag("graphml"))
    add_graphml_keys(root, dimension)
    graph = ET.SubElement(root, tag("graph"), {"id": "patacon_trace", "edgedefault": "directed"})
    max_step = max(len(path) - 1 for _, path in paths)
    max_display_step = max(0, max_step)
    solution_order: list[str] = []
    slot_steps: list[dict[str, Any]] = []

    data(graph, GRAPH_KEYS["planner"], str(result.get("planner", "PATACON")))
    data(graph, GRAPH_KEYS["trace_level"], "nodes")
    data(graph, GRAPH_KEYS["dimension"], dimension)
    data(graph, GRAPH_KEYS["max_grow_step"], max_display_step)
    data(graph, GRAPH_KEYS["max_display_step"], max_display_step)
    data(graph, GRAPH_KEYS["max_parallel_step"], max(0, len(paths) - 1))
    data(graph, GRAPH_KEYS["joint_names"], json.dumps(joint_names))
    solution_order_data = data(graph, GRAPH_KEYS["solution_order"], "[]")
    slot_steps_data = data(graph, GRAPH_KEYS["slot_steps_json"], "[]")
    data(graph, GRAPH_KEYS["timeline_events_json"], "[]")

    node_seq = 0
    edge_seq = 0
    for path_idx, (meta, path) in enumerate(paths):
        candidate_idx = int(meta.get("candidate_idx", path_idx))
        accepted = bool(meta.get("accepted", False))
        path_node_ids = [f"n_generated_path_{candidate_idx}_{state_idx}" for state_idx in range(len(path))]
        if accepted:
            solution_order = path_node_ids
        for state_idx, q in enumerate(path):
            node_id = path_node_ids[state_idx]
            node = ET.SubElement(graph, tag("node"), {"id": node_id})
            node_values = {
                "seq": node_seq,
                "tree": "generated_path",
                "batch_idx": candidate_idx,
                "node_idx": state_idx,
                "parent_idx": state_idx - 1,
                "parent_id": "" if state_idx == 0 else path_node_ids[state_idx - 1],
                "iter": int(meta.get("iter", state_idx)),
                "phase": "generated_path",
                "step_type": "connect",
                "slot_idx": candidate_idx,
                "escape_step": -1,
                "ts_id": -1,
                "is_proj_root": state_idx == 0,
                "grow_step": state_idx,
                "duration_sec": 0.0,
                "display_step": state_idx,
                "parallel_step": path_idx,
                "parallel_step_start_sec": 0.0,
                "parallel_step_finished_at_sec": 0.0,
                "parallel_step_duration_sec": 0.0,
                "depth": state_idx,
                "solution": True,
                "event_kind": "node_add",
                "active": 1,
                "advanced": 1 if state_idx > 0 else 0,
                "trapped": 0,
                "reached": 1 if state_idx == len(path) - 1 else 0,
                "mean_progress": state_idx / max(1, len(path) - 1),
                "max_progress": state_idx / max(1, len(path) - 1),
                "simultaneous": len(paths) > 1,
                "simultaneous_group": f"generated_path_{candidate_idx}",
                "order_index": state_idx if accepted else -1,
            }
            for key, value in node_values.items():
                data(node, NODE_KEYS[key], value)
            data(node, NODE_KEYS["q_json"], json.dumps(q))
            for q_idx, value in enumerate(q):
                data(node, f"n_q{q_idx}", value)
            node_seq += 1

        for state_idx in range(1, len(path)):
            edge = ET.SubElement(
                graph,
                tag("edge"),
                {
                    "id": f"e_generated_path_{candidate_idx}_{state_idx - 1}_{edge_seq}",
                    "source": path_node_ids[state_idx - 1],
                    "target": path_node_ids[state_idx],
                },
            )
            edge_values = {
                "kind": "solution",
                "tree": "generated_path",
                "batch_idx": candidate_idx,
                "iter": int(meta.get("iter", state_idx)),
                "phase": "generated_path",
                "grow_step": state_idx,
                "display_step": state_idx,
                "parallel_step": path_idx,
                "solution": True,
            }
            for key, value in edge_values.items():
                data(edge, EDGE_KEYS[key], value)
            slot_steps.append(
                {
                    "seq": edge_seq,
                    "iter": int(meta.get("iter", state_idx)),
                    "tree": "generated_path",
                    "batch_idx": candidate_idx,
                    "slot_idx": candidate_idx,
                    "phase": "generated_path",
                    "step": "path_edge",
                    "step_type": "connect",
                    "result": "advanced",
                    "substep": 0,
                    "grow_step": state_idx,
                    "display_step": state_idx,
                    "parallel_step": path_idx,
                    "duration_sec": 0.0,
                    "progress": state_idx / max(1, len(path) - 1),
                }
            )
            edge_seq += 1

    if not solution_order and paths:
        solution_order = [f"n_generated_path_{int(paths[0][0].get('candidate_idx', 0))}_{idx}" for idx in range(len(paths[0][1]))]

    solution_order_data.text = json.dumps(solution_order)
    slot_steps_data.text = json.dumps(slot_steps)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True), len(paths), node_seq


def add_aorrtc_update_colors(html_path: Path) -> None:
    """Color previous AORRTC solution overlays without breaking export.

    add_pca_3d_layout() contains a *reference* to aorrtcUpdateColor before the
    function is actually defined, so idempotence must check for the function
    declaration itself rather than the bare string ``aorrtcUpdateColor``.
    """
    html = html_path.read_text(encoding="utf-8")
    if "function aorrtcUpdateColor(index)" in html:
        return

    draw_marker = "function draw() {"
    stroke_markers = [
        'ctx.strokeStyle = edge.kind === "connection" ? "#c084fc" '
        ': edge.solution ? "#4ade80" :',
        'ctx.strokeStyle = edge.kind === "connection" ? "#c084fc" '
        ': edge.solution ? "#15803d" :',
    ]
    stroke_marker = next((marker for marker in stroke_markers if marker in html), None)
    width_marker = (
        'ctx.lineWidth = edge.solution ? 4.2 '
        ': edge.kind === "connection" ? 3.6 : 1.15;'
    )

    # Coloring is cosmetic. Never fail the whole trace export merely because
    # the upstream PATACON viewer changed an exact drawing string.
    if draw_marker not in html or stroke_marker is None or width_marker not in html:
        return

    color_script = """
function aorrtcUpdateColor(index) {
  const safeIndex = Math.max(0, Number(index) || 0);
  const hue = (safeIndex * 137.508) % 360;
  return `hsla(${hue}, 85%, 48%, 0.88)`;
}
"""
    html = html.replace(draw_marker, color_script + "\n" + draw_marker, 1)
    html = html.replace(
        stroke_marker,
        'ctx.strokeStyle = edge.kind === "solution_update" ? aorrtcUpdateColor(edge.batch_idx) '
        ': edge.kind === "connection" ? "#c084fc" '
        ': edge.solution ? "#15803d" :',
        1,
    )
    html = html.replace(
        width_marker,
        'ctx.lineWidth = edge.kind === "solution_update" ? 2.8 '
        ': edge.solution ? 4.2 '
        ': edge.kind === "connection" ? 3.6 : 1.15;',
        1,
    )
    dash_marker = (
        'if (edge.kind === "connection") ctx.setLineDash([5,4]); '
        'else ctx.setLineDash([]);'
    )
    if dash_marker in html:
        html = html.replace(
            dash_marker,
            'if (edge.kind === "solution_update") ctx.setLineDash([7,4]); '
            'else if (edge.kind === "connection") ctx.setLineDash([5,4]); '
            'else ctx.setLineDash([]);',
            1,
        )
    html_path.write_text(html, encoding="utf-8")

def add_invalid_configuration_guard(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    if "validPcaConfiguration" in html:
        return

    compute_marker = "function computePca(nodes) {"
    samples_marker = "const samples = nodes.filter(n => n.q && n.q.length);"
    projection_marker = (
        "  for (const n of nodes) {\n"
        "    const centered = n.q.map((x,i) => (x || 0) - (mean[i] || 0));"
    )
    visible_marker = "function visibleNode(n) {"
    if any(
        marker not in html
        for marker in (
            compute_marker,
            samples_marker,
            projection_marker,
            visible_marker,
        )
    ):
        raise RuntimeError(
            "PATACON HTML viewer is incompatible with PCA configuration validation"
        )

    guard_script = """
function validPcaConfiguration(n) {
  return Array.isArray(n.q) && n.q.length > 0 && n.q.every(value =>
    Number.isFinite(value) && Math.abs(value - (-9999.0)) > 1.0e-3
  );
}
"""
    html = html.replace(
        compute_marker,
        guard_script + "\n" + compute_marker,
        1,
    )
    html = html.replace(
        samples_marker,
        (
            "const samples = nodes.filter(n => "
            "validPcaConfiguration(n) && n.tree !== \"history\");"
        ),
        1,
    )
    html = html.replace(
        projection_marker,
        "  for (const n of nodes) {\n"
        "    if (!validPcaConfiguration(n)) { n.pca = null; continue; }\n"
        "    const centered = n.q.map((x,i) => (x || 0) - (mean[i] || 0));",
        1,
    )
    html = html.replace(
        visible_marker,
        visible_marker
        + "\n\t  if (n.tree === \"history\") return false;"
        + "\n\t  if (layoutSelect.value === \"pca\" && !n.pca) return false;",
        1,
    )
    html_path.write_text(html, encoding="utf-8")


def add_aorrtc_history_visibility(html_path: Path) -> None:
    """Keep AORRTC history edges visible while hiding synthetic history nodes.

    The synthetic nodes are only coordinate carriers for previous solution
    trajectories. They must not appear as tree nodes, but their edges still
    need to be rendered. This patch also includes history coordinates in the
    2D viewport bounds so an old path is not clipped when it lies outside the
    final/best tree extent.
    """
    html = html_path.read_text(encoding="utf-8")
    patch_marker = "/* PATACON AORRTC history visibility */"

    # The base PATACON viewer only creates startTree/goalTree checkboxes.
    # Synthetic history nodes use tree="history". Without this guard,
    # updateSlotPanel() calls activeTree("history") and crashes on
    # document.getElementById("historyTree") == null before any edges draw.
    active_tree_old = (
        'function activeTree(tree) { return document.getElementById(tree + "Tree").checked; }'
    )
    active_tree_new = (
        'function activeTree(tree) {\n'
        '  if (tree === "history") return false;\n'
        '  const control = document.getElementById(tree + "Tree");\n'
        '  return control ? control.checked : false;\n'
        '}'
    )
    if active_tree_old in html:
        html = html.replace(active_tree_old, active_tree_new, 1)

    if patch_marker in html:
        html_path.write_text(html, encoding="utf-8")
        return

    visible_start = html.find("function visibleEdge(e) {")
    next_function = html.find("function countSlotModes", visible_start)
    if visible_start < 0 or next_function < 0:
        raise RuntimeError(
            "PATACON HTML viewer is incompatible with AORRTC history visibility"
        )

    replacement = r'''/* PATACON AORRTC history visibility */
function visibleEdge(e) {
  const a = nodeById.get(e.source);
  const b = nodeById.get(e.target);
  if (!a || !b) return false;

  // Previous AORRTC solutions use hidden synthetic nodes. Their path edges
  // remain visible even though visibleNode(historyNode) intentionally returns
  // false. The slider still controls when the overlay appears.
  if (e.kind === "solution_update") {
    return itemStep(e) <= Number(slider.value);
  }

  return visibleNode(a)
      && visibleNode(b)
      && itemStep(e) <= Number(slider.value);
}
'''
    html = html[:visible_start] + replacement + html[next_function:]

    old_bounds = (
        "  const shown = trace.nodes.filter(visibleNode);\n"
        "  const points = (shown.length ? shown : trace.nodes).map(coordinates);"
    )
    new_bounds = (
        "  const shown = trace.nodes.filter(visibleNode);\n"
        "  const historyShown = trace.nodes.filter(n =>\n"
        "    n.tree === \"history\" &&\n"
        "    itemStep(n) <= Number(slider.value) &&\n"
        "    (layoutSelect.value !== \"pca\" || !!n.pca)\n"
        "  );\n"
        "  const boundNodes = [...shown, ...historyShown];\n"
        "  const points = (boundNodes.length ? boundNodes : trace.nodes).map(coordinates);"
    )
    if old_bounds in html:
        html = html.replace(old_bounds, new_bounds, 1)

    html_path.write_text(html, encoding="utf-8")


def add_pca_3d_layout(html_path: Path) -> None:
    """Add a perspective PC1/PC2/PC3 tree viewer to PATACON HTML.

    This is a visualization-only post-process. The planner trace and CUDA
    execution are unchanged. PCA 2D and Timeline modes remain intact; the
    added PCA 3D mode uses a dedicated perspective renderer with orbit,
    zoom, grid, axes, depth sorting, and explained-variance readout.
    """
    html = html_path.read_text(encoding="utf-8")
    patch_marker = "/* PATACON PCA 3D perspective viewer */"
    if patch_marker in html:
        return

    option_marker = '<option value="pca">PCA layout</option>'
    load_marker = "computePca(trace.nodes);"
    visible_marker = "function visibleNode(n) {"
    body_end_marker = "</body>"
    if any(
        marker not in html
        for marker in (option_marker, load_marker, visible_marker, body_end_marker)
    ):
        raise RuntimeError(
            "PATACON HTML viewer is incompatible with the PCA 3D perspective patch"
        )

    html = html.replace(
        option_marker,
        option_marker + '<option value="pca3d">PCA 3D layout</option>',
        1,
    )

    pca3d_compute_script = r'''
/* PATACON PCA 3D perspective viewer */
const pataconPca3dStats = {
  eigenvalues: [0, 0, 0],
  explained: [0, 0, 0],
  radius: 1,
};

function computePca3d(nodes) {
  const samples = nodes.filter(
    n => validPcaConfiguration(n) && n.tree !== "history"
  );
  if (!samples.length) return;

  const d = samples[0].q.length;
  if (d <= 0) return;

  const mean = Array(d).fill(0);
  for (const n of samples) {
    for (let i = 0; i < d; ++i) mean[i] += n.q[i];
  }
  for (let i = 0; i < d; ++i) mean[i] /= samples.length;

  function dot(a, b) {
    let value = 0;
    for (let i = 0; i < d; ++i) value += (a[i] || 0) * (b[i] || 0);
    return value;
  }

  function normalize(v) {
    let norm2 = 0;
    for (const value of v) norm2 += value * value;
    const norm = Math.sqrt(norm2);
    if (!(norm > 1.0e-12)) return Array(d).fill(0);
    return v.map(value => value / norm);
  }

  function covMul(v) {
    const out = Array(d).fill(0);
    for (const n of samples) {
      let projected = 0;
      for (let i = 0; i < d; ++i) {
        projected += (n.q[i] - mean[i]) * v[i];
      }
      for (let i = 0; i < d; ++i) {
        out[i] += (n.q[i] - mean[i]) * projected;
      }
    }
    const denom = Math.max(1, samples.length - 1);
    return out.map(value => value / denom);
  }

  function seedVector(axis) {
    const seed = Array(d).fill(0);
    const primary = Math.min(axis, d - 1);
    seed[primary] = 1.0;
    for (let i = 0; i < d; ++i) {
      if (i !== primary) seed[i] = 0.013 / (i + axis + 2);
    }
    return normalize(seed);
  }

  function dominantComponent(axis, previous) {
    let v = seedVector(axis);
    for (let iteration = 0; iteration < 60; ++iteration) {
      let next = covMul(v);
      for (const basis of previous) {
        const amount = dot(next, basis);
        next = next.map((value, i) => value - amount * basis[i]);
      }
      const normalized = normalize(next);
      if (!normalized.some(value => Math.abs(value) > 1.0e-12)) break;
      v = normalized;
    }
    return v;
  }

  const v1 = dominantComponent(0, []);
  const v2 = dominantComponent(1, [v1]);
  const v3 = dominantComponent(2, [v1, v2]);

  const cv1 = covMul(v1), cv2 = covMul(v2), cv3 = covMul(v3);
  const l1 = Math.max(0, dot(v1, cv1));
  const l2 = Math.max(0, dot(v2, cv2));
  const l3 = Math.max(0, dot(v3, cv3));

  let totalVariance = 0;
  const denom = Math.max(1, samples.length - 1);
  for (const n of samples) {
    for (let i = 0; i < d; ++i) {
      const delta = n.q[i] - mean[i];
      totalVariance += delta * delta / denom;
    }
  }
  const safeTotal = Math.max(totalVariance, 1.0e-12);
  pataconPca3dStats.eigenvalues = [l1, l2, l3];
  pataconPca3dStats.explained = [l1 / safeTotal, l2 / safeTotal, l3 / safeTotal];

  let radius = 0;
  for (const n of nodes) {
    if (!validPcaConfiguration(n)) {
      n.pca3d = null;
      continue;
    }
    const centered = n.q.map((value, i) => value - mean[i]);
    n.pca3d = [dot(centered, v1), dot(centered, v2), dot(centered, v3)];
    radius = Math.max(radius, Math.hypot(...n.pca3d));
  }
  pataconPca3dStats.radius = Math.max(radius, 1.0e-9);
}
'''

    html = html.replace(
        visible_marker,
        pca3d_compute_script + "\n" + visible_marker,
        1,
    )
    html = html.replace(
        load_marker,
        load_marker + "\n  computePca3d(trace.nodes);",
        1,
    )

    interaction_script = r'''
<script>
(() => {
  "use strict";

  const pca3d = {
    yaw: -0.72,
    pitch: 0.48,
    zoom: 1.0,
    drag: null,
    screenNodes: [],
  };

  const originalDraw = draw;
  const originalResetView = resetView;

  function pca3dActive() {
    return layoutSelect.value === "pca3d";
  }

  function normalizedWorld(raw) {
    const radius = Math.max(pataconPca3dStats.radius, 1.0e-9);
    return [raw[0] / radius, raw[1] / radius, raw[2] / radius];
  }

  function cameraTransformWorld(world) {
    const [x, y, z] = world;
    const cy = Math.cos(pca3d.yaw), sy = Math.sin(pca3d.yaw);
    const cp = Math.cos(pca3d.pitch), sp = Math.sin(pca3d.pitch);
    const x1 = cy * x + sy * z;
    const z1 = -sy * x + cy * z;
    const y2 = cp * y - sp * z1;
    const z2 = sp * y + cp * z1;
    return [x1, y2, z2];
  }

  function cameraTransformRaw(raw) {
    return cameraTransformWorld(normalizedWorld(raw));
  }

  function projectWorld(world) {
    const [x, y, z] = cameraTransformWorld(world);
    const cameraDistance = 4.2;
    const denom = Math.max(0.35, cameraDistance - z);
    const focal = Math.min(width, height) * 1.55 * pca3d.zoom;
    return {
      x: width * 0.5 + x * focal / denom,
      y: height * 0.51 - y * focal / denom,
      z,
      depth: denom,
      scale: focal / denom,
    };
  }

  function projectRaw(raw) {
    return projectWorld(normalizedWorld(raw));
  }

  function line3d(a, b, stroke, lineWidth=1, dash=[]) {
    const pa = projectWorld(a), pb = projectWorld(b);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function label3d(text, p, fill, dx=5, dy=-5) {
    const s = projectWorld(p);
    ctx.fillStyle = fill;
    ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(text, s.x + dx, s.y + dy);
  }

  function drawGridAndAxes() {
    const gridExtent = 1.18;
    const gridStep = 0.2;
    for (let v = -1.0; v <= 1.0001; v += gridStep) {
      const major = Math.abs(v) < 1.0e-6;
      const stroke = major ? "rgba(100,116,139,.48)" : "rgba(148,163,184,.22)";
      line3d([-gridExtent, 0, v], [gridExtent, 0, v], stroke, major ? 1.15 : 0.8);
      line3d([v, 0, -gridExtent], [v, 0, gridExtent], stroke, major ? 1.15 : 0.8);
    }

    const axis = 1.34;
    line3d([-axis,0,0], [axis,0,0], "#dc2626", 2.2);
    line3d([0,-axis,0], [0,axis,0], "#16a34a", 2.2);
    line3d([0,0,-axis], [0,0,axis], "#2563eb", 2.2);
    label3d("PC1", [axis,0,0], "#991b1b", 7, -3);
    label3d("PC2", [0,axis,0], "#166534", 7, -3);
    label3d("PC3", [0,0,axis], "#1d4ed8", 7, -3);
  }

  function edgeDepth(edge) {
    const a = nodeById.get(edge.source), b = nodeById.get(edge.target);
    if (!a?.pca3d || !b?.pca3d) return -Infinity;
    return (cameraTransformRaw(a.pca3d)[2] + cameraTransformRaw(b.pca3d)[2]) * 0.5;
  }

  function nodeColor(n, solution) {
    if (solution) return "#15803d";
    return n.tree === "start" ? "#0284c7" : "#ea580c";
  }

  function drawPca3d() {
    drawPending = false;
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,width,height);
    if (!trace) return;

    const currentStep = Number(slider.value);
    stepValue.textContent = String(slider.value);
    const visibleNodesNow = trace.nodes.filter(n => visibleNode(n) && !!n.pca3d);
    updateStats(visibleNodesNow);
    updateSlotPanel();
    updateAsyncPanel();
    pca3d.screenNodes = [];
    screenNodes = [];

    drawGridAndAxes();

    const edges = trace.edges
      .filter(edge => visibleEdge(edge))
      .filter(edge => nodeById.get(edge.source)?.pca3d && nodeById.get(edge.target)?.pca3d)
      .sort((a,b) => edgeDepth(a) - edgeDepth(b));

    ctx.lineCap = "round";
    for (const edge of edges) {
      const a = nodeById.get(edge.source), b = nodeById.get(edge.target);
      const pa = projectRaw(a.pca3d), pb = projectRaw(b.pca3d);
      const historyEdge = edge.kind === "solution_update";
      const highlighted = edge.solution || edge.kind === "connection" || historyEdge;
      const meanDepth = (pa.depth + pb.depth) * 0.5;
      const perspective = Math.max(0.62, Math.min(1.35, 4.2 / meanDepth));

      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.strokeStyle = historyEdge && typeof aorrtcUpdateColor === "function"
        ? aorrtcUpdateColor(edge.batch_idx)
        : edge.kind === "connection"
          ? "#7e22ce"
          : edge.solution
            ? "#15803d"
            : a.tree === "start"
              ? "rgba(2,132,199,.42)"
              : "rgba(234,88,12,.42)";
      const baseWidth = historyEdge ? 2.6 : highlighted ? 3.2 : 1.05;
      ctx.lineWidth = baseWidth * perspective;
      if (historyEdge) ctx.setLineDash([7,4]);
      else if (edge.kind === "connection") ctx.setLineDash([6,4]);
      else ctx.setLineDash([]);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    const nodes = visibleNodesNow
      .map(n => ({n, p: projectRaw(n.pca3d)}))
      .sort((a,b) => b.p.depth - a.p.depth);

    for (const item of nodes) {
      const n = item.n, p = item.p;
      const solution = n.solution || solutionIds.has(n.id);
      const root = n.parent_idx < 0;
      const current = itemStep(n) === currentStep;
      const perspective = Math.max(0.68, Math.min(1.5, 4.2 / p.depth));
      const baseRadius = solution ? 5.6 : root ? 5.0 : current ? 4.2 : 3.2;
      const r = baseRadius * perspective;
      const alpha = Math.max(0.42, Math.min(1.0, 1.20 - (p.depth - 3.2) * 0.20));

      ctx.globalAlpha = solution ? 1.0 : alpha;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = nodeColor(n, solution);
      ctx.fill();
      if (root || solution || current) {
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = Math.max(0.8, 1.0 * perspective);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      const hitRadius = Math.max(7, r + 2);
      pca3d.screenNodes.push({n, x:p.x, y:p.y, r:hitRadius, depth:p.depth});
      screenNodes.push({n, x:p.x, y:p.y, r:hitRadius});
    }

    const e = pataconPca3dStats.explained.map(value => (100 * value).toFixed(1));
    const p123 = 100 * pataconPca3dStats.explained.reduce((a,b) => a+b, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(255,255,255,.90)";
    ctx.fillRect(14, 14, 300, 72);
    ctx.strokeStyle = "rgba(100,116,139,.55)";
    ctx.lineWidth = 1;
    ctx.strokeRect(14.5, 14.5, 299, 71);
    ctx.fillStyle = "#0f172a";
    ctx.font = "700 12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("PCA 3D perspective", 25, 35);
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(`PC1 ${e[0]}%   PC2 ${e[1]}%   PC3 ${e[2]}%`, 25, 54);
    ctx.fillText(`PC1-3 ${p123.toFixed(1)}%   drag: orbit   wheel: zoom`, 25, 70);
  }

  draw = function() {
    if (pca3dActive()) drawPca3d();
    else originalDraw();
  };

  resetView = function() {
    if (pca3dActive()) {
      pca3d.yaw = -0.72;
      pca3d.pitch = 0.48;
      pca3d.zoom = 1.0;
      view = {x:0,y:0,k:1};
      calculateBounds();
      schedule();
      return;
    }
    originalResetView();
  };

  canvas.addEventListener("pointerdown", event => {
    if (!pca3dActive()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    pca3d.drag = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      yaw: pca3d.yaw,
      pitch: pca3d.pitch,
    };
    canvas.setPointerCapture(event.pointerId);
  }, true);

  canvas.addEventListener("pointermove", event => {
    if (!pca3dActive()) return;
    event.preventDefault();
    event.stopImmediatePropagation();

    if (pca3d.drag && event.pointerId === pca3d.drag.pointerId) {
      pca3d.yaw = pca3d.drag.yaw + (event.clientX - pca3d.drag.x) * 0.008;
      pca3d.pitch = Math.max(
        -1.48,
        Math.min(1.48, pca3d.drag.pitch + (event.clientY - pca3d.drag.y) * 0.008),
      );
      tooltip.style.display = "none";
      schedule();
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    let hit = null, best = Infinity;
    for (const item of pca3d.screenNodes) {
      const dist2 = (item.x - x) ** 2 + (item.y - y) ** 2;
      const score = dist2 + Math.max(0, item.depth) * 1.0e-4;
      if (dist2 <= item.r ** 2 && score < best) {
        best = score;
        hit = item.n;
      }
    }
    if (!hit) {
      tooltip.style.display = "none";
      return;
    }

    const pc = hit.pca3d || [0,0,0];
    const labels = trace.jointNames.length === hit.q.length
      ? trace.jointNames.map((name,i) => `${name}=${hit.q[i].toFixed(4)}`).join("\n")
      : hit.q.map((v,i) => `q${i}=${v.toFixed(4)}`).join("\n");
    tooltip.textContent =
      `${hit.tree}[${hit.node_idx}] ${hit.phase}\n` +
      `PC1=${pc[0].toFixed(5)}  PC2=${pc[1].toFixed(5)}  PC3=${pc[2].toFixed(5)}\n` +
      `display_step=${hit.display_step} parallel_step=${hit.parallel_step} depth=${hit.depth}\n` +
      labels;
    tooltip.style.display = "block";
    tooltip.style.left = Math.min(innerWidth - 640, event.clientX + 13) + "px";
    tooltip.style.top = Math.min(innerHeight - 240, event.clientY + 13) + "px";
  }, true);

  function finishOrbit(event) {
    if (!pca3dActive() || !pca3d.drag) return;
    if (event.pointerId !== pca3d.drag.pointerId) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    pca3d.drag = null;
  }
  canvas.addEventListener("pointerup", finishOrbit, true);
  canvas.addEventListener("pointercancel", finishOrbit, true);

  canvas.addEventListener("wheel", event => {
    if (!pca3dActive()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    pca3d.zoom = Math.max(0.25, Math.min(6.0, pca3d.zoom * Math.exp(-event.deltaY * 0.001)));
    schedule();
  }, {capture:true, passive:false});

  layoutSelect.addEventListener("change", () => {
    tooltip.style.display = "none";
    if (pca3dActive()) {
      pca3d.yaw = -0.72;
      pca3d.pitch = 0.48;
      pca3d.zoom = 1.0;
    }
    calculateBounds();
    schedule();
  });

  if (typeof trace !== "undefined" && trace && Array.isArray(trace.nodes)) {
    computePca3d(trace.nodes);
  }
})();
</script>
'''

    html = html.replace(
        body_end_marker,
        interaction_script + "\n" + body_end_marker,
        1,
    )
    html_path.write_text(html, encoding="utf-8")


def add_white_canvas_theme(html_path: Path) -> None:
    """Apply the white canvas theme as a best-effort visualization patch.

    This must never abort trace export. AORRTC color post-processing can alter
    exact PATACON color strings, so every JavaScript recolor replacement is
    optional. The CSS canvas override is sufficient for the base theme.
    """
    html = html_path.read_text(encoding="utf-8")
    theme_marker = "/* PATACON white trace canvas */"
    if theme_marker in html:
        return

    style_end_marker = "</style>"
    if style_end_marker not in html:
        return

    html = html.replace(
        style_end_marker,
        (
            f"{theme_marker}\n"
            "#canvas{background:#fff}\n"
            "#slotPanel{display:none}\n"
            f"{style_end_marker}"
        ),
        1,
    )

    replacements = [
        (
            'if (root || solution || current) { ctx.strokeStyle = "#f8fafc";',
            'if (root || solution || current) { ctx.strokeStyle = "#0f172a";',
        ),
        (
            'ctx.strokeStyle = halo || (n.simultaneous ? "#f8fafc" : "#94a3b8");',
            'ctx.strokeStyle = halo || (n.simultaneous ? "#0f172a" : "#64748b");',
        ),
        (
            'a.tree === "start" ? "rgba(56,189,248,.38)" : "rgba(251,146,60,.38)"',
            'a.tree === "start" ? "rgba(3,105,161,.72)" : "rgba(194,65,12,.72)"',
        ),
        (
            'ctx.fillStyle = solution ? "#4ade80" : n.tree === "start" ? "#38bdf8" : "#fb923c";',
            'ctx.fillStyle = solution ? "#15803d" : n.tree === "start" ? "#0284c7" : "#ea580c";',
        ),
        (
            ': edge.solution ? "#4ade80" :',
            ': edge.solution ? "#15803d" :',
        ),
        (
            'style="background:#4ade80"></span>solution chain',
            'style="background:#15803d"></span>solution chain',
        ),
    ]
    for old, new in replacements:
        if old in html:
            html = html.replace(old, new, 1)

    html_path.write_text(html, encoding="utf-8")

def save_patacon_html(graphml: str, html_path: Path, patacon_root: Path, title: str) -> None:
    exporter = patacon_root / "patacon" / "planner" / "tbrrt" / "trace_graphml.py"
    if not exporter.is_file():
        raise FileNotFoundError(
            "PATACON HTML exporter not found under "
            f"{patacon_root}; pass --patacon-root or set PATACON_ROOT"
        )
    sys.path.insert(0, str(patacon_root))
    from patacon.planner.tbrrt.trace_graphml import save_trace_html

    save_trace_html(graphml, html_path, title=title)
    add_invalid_configuration_guard(html_path)
    add_pca_3d_layout(html_path)
    if "solution_update" in graphml:
        add_aorrtc_history_visibility(html_path)
    # Apply the base theme before update-color rewriting so exact upstream
    # PATACON color markers are still available.
    add_white_canvas_theme(html_path)
    if "solution_update" in graphml or "solution_final" in graphml:
        add_aorrtc_update_colors(html_path)


def default_output_path(result_json: Path, suffix: str) -> Path:
    stem = result_json.with_suffix("")
    return stem.parent / f"{stem.name}{suffix}"


def default_patacon_root(repo_root: Path) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("PATACON_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            repo_root.parent / "patacon",
            Path.home() / "gh_ws" / "tb_rrt_ws" / "src" / "patacon",
        )
    )
    for candidate in candidates:
        exporter = (
            candidate
            / "patacon"
            / "planner"
            / "tbrrt"
            / "trace_graphml.py"
        )
        if exporter.is_file():
            return candidate
    return candidates[0]


def print_missing_result_help(result_json: Path, trace_mode: str = "auto") -> None:
    print(f"missing result JSON: {result_json}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Create it first, for example:", file=sys.stderr)
    if trace_mode == "tree" or "tree" in result_json.name:
        print(
            "  ./build/single_mbm ffw_sg2 tray_lift 1 "
            "--no-print-path --trace-trees "
            f"--save-json {result_json}",
            file=sys.stderr,
        )
    else:
        print(
            "  ./build/single_mbm ffw_sg2 tray_lift 1 "
            f"--save-json {result_json}",
            file=sys.stderr,
        )

    trace_dir = result_json.parent
    if trace_dir.exists():
        candidates = sorted(trace_dir.glob("*result*.json"))
        if candidates:
            print("", file=sys.stderr)
            print("Existing result JSON candidates:", file=sys.stderr)
            for candidate in candidates[:10]:
                print(f"  {candidate}", file=sys.stderr)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert a saved PATACON path JSON into GraphML/HTML trace files."
    )
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--result-index", type=int, default=0)
    parser.add_argument("--trace-mode", choices=("auto", "tree", "path", "paths"), default="auto")
    parser.add_argument("--path-key", default="path_start_to_goal")
    parser.add_argument("--graphml", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument(
        "--html-trace-mode",
        choices=("path", "tree", "paths"),
        default="path",
        help=(
            "Visualization mode embedded in the HTML. The GraphML output still "
            "uses --trace-mode. Default path keeps the HTML lightweight."
        ),
    )
    parser.add_argument(
        "--html-max-tree-nodes",
        type=int,
        default=6000,
        help=(
            "Maximum tree nodes embedded in tree-mode HTML. The full GraphML "
            "file is still saved. Use 0 to embed every tree node in HTML."
        ),
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=0,
        help="Maximum generated paths embedded for --trace-mode paths; 0 means all stored paths.",
    )
    parser.add_argument(
        "--patacon-root",
        type=Path,
        default=default_patacon_root(repo_root),
        help="PATACON repository root used for the HTML viewer implementation.",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    result_json = args.result_json.expanduser().resolve()
    if not result_json.exists():
        print_missing_result_help(result_json, args.trace_mode)
        return 1

    result = load_result(result_json, args.result_index)
    use_paths_trace = args.trace_mode == "paths"
    use_tree_trace = args.trace_mode == "tree" or (
        args.trace_mode == "auto" and has_tree_trace(result)
    )
    if use_paths_trace:
        graphml, path_count, state_count = generated_paths_graphml_text(
            result,
            max_paths=args.max_paths if args.max_paths > 0 else None,
        )
        trace_mode = "paths"
    elif use_tree_trace:
        if not has_tree_trace(result):
            raise ValueError("requested --trace-mode tree, but the result JSON has no tree_trace")
        graphml = tree_trace_graphml_text(result)
        trace_mode = "tree"
        state_count = sum(
            len(tree.get("nodes", []))
            for tree in result.get("tree_trace", {}).get("trees", [])
            if isinstance(tree, dict)
        )
    else:
        path = normalized_path(result, args.path_key)
        graphml = graphml_text(result, path)
        trace_mode = "path"
        state_count = len(path)

    graphml_path = (args.graphml or default_output_path(result_json, "_trace.graphml")).expanduser()
    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    graphml_path.write_text(graphml, encoding="utf-8")
    print(f"saved_graphml: {graphml_path}")

    if not args.no_html:
        html_path = (args.html or default_output_path(result_json, "_trace.html")).expanduser()
        title = args.title or (
            f"PATACON {result.get('robot', '')} {result.get('problem_name', '')} path trace"
        ).strip()
        if args.html_trace_mode == "path":
            if use_paths_trace:
                html_graphml = graphml
            else:
                html_path_states = normalized_path(result, args.path_key)
                html_graphml = graphml_text(result, html_path_states)
        elif args.html_trace_mode == "paths":
            html_graphml, _, _ = generated_paths_graphml_text(
                result,
                max_paths=args.max_paths if args.max_paths > 0 else None,
            )
        elif use_tree_trace and args.html_max_tree_nodes > 0:
            html_graphml = tree_trace_graphml_text(result, max_tree_nodes=args.html_max_tree_nodes)
        else:
            html_graphml = graphml
        save_patacon_html(html_graphml, html_path, args.patacon_root.expanduser().resolve(), title)
        print(f"saved_html: {html_path}")
        print(f"html_trace_mode: {args.html_trace_mode}")
        if args.html_trace_mode == "tree" and use_tree_trace and args.html_max_tree_nodes > 0:
            print(f"html_tree_nodes_limit: {args.html_max_tree_nodes}")

    print(f"trace_mode: {trace_mode}")
    print(f"states: {state_count}")
    if trace_mode == "paths":
        print(f"paths: {path_count}")
    print(f"path_key: {args.path_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
