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
    solution_order_by_id: dict[str, int],
    max_tree_nodes: int | None,
) -> dict[str, dict[int, dict[str, Any]]]:
    if max_tree_nodes is None or max_tree_nodes <= 0:
        return nodes_by_tree

    total_nodes = sum(len(nodes) for nodes in nodes_by_tree.values())
    if total_nodes <= max_tree_nodes:
        return nodes_by_tree

    selected_by_tree: dict[str, set[int]] = {tree: set() for tree in nodes_by_tree}
    for node_id in solution_order_by_id:
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
    ready_nodes_by_tree = compact_tree_nodes(
        ready_nodes_by_tree,
        solution_order_by_id,
        max_tree_nodes,
    )

    data(graph, GRAPH_KEYS["planner"], str(result.get("planner", "pRRTC")))
    data(graph, GRAPH_KEYS["trace_level"], "nodes")
    data(graph, GRAPH_KEYS["dimension"], dimension)
    data(graph, GRAPH_KEYS["max_grow_step"], max_step)
    data(graph, GRAPH_KEYS["max_display_step"], max_step)
    data(graph, GRAPH_KEYS["max_parallel_step"], max_step)
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
                "step_type": "connect" if order_index >= 0 else "extend",
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
                "solution": order_index >= 0,
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

    data(graph, GRAPH_KEYS["planner"], str(result.get("planner", "pRRTC")))
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
    raw_paths = result.get("generated_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("result does not contain generated_paths; rerun with --aorrtc and --trace-mode paths")

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

    data(graph, GRAPH_KEYS["planner"], str(result.get("planner", "pRRTC")))
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
        description="Convert a saved pRRTC path JSON into PATACON-style GraphML/HTML trace files."
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
            f"pRRTC {result.get('robot', '')} {result.get('problem_name', '')} path trace"
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
