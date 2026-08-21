"""Deterministic evaluation for structured MuseKG question-answer pairs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx


def normalise_text(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().strip("'\"").casefold().split())


def _as_list(value: Any) -> list[str | None]:
    if isinstance(value, list):
        return [normalise_text(item) for item in value]
    return [normalise_text(value)] if value is not None else []


def answers_equal(expected: Any, actual: Any) -> bool:
    return set(_as_list(expected)) == set(_as_list(actual))


def _find_object(graph: nx.Graph, title: str) -> str | None:
    wanted = normalise_text(title)
    for node, data in graph.nodes(data=True):
        if data.get("type") == "object" and normalise_text(data.get("title")) == wanted:
            return node
    return None


def _edge_has_relationship(graph: nx.Graph, source: str, target: str, relationship: str) -> bool:
    edge_data = graph.get_edge_data(source, target) or {}
    wanted = normalise_text(relationship)
    if graph.is_multigraph():
        return any(
            normalise_text(attributes.get("relationship")) == wanted
            for attributes in edge_data.values()
        )
    return normalise_text(edge_data.get("relationship")) == wanted


def _display_name(graph: nx.Graph, node: str) -> str:
    data = graph.nodes[node]
    return str(data.get("title") or data.get("label") or node)


def query_graph(graph: nx.Graph, query_type: str, details: dict[str, Any]) -> Any:
    if query_type == "find_attribute":
        node = _find_object(graph, details.get("object_title", ""))
        return graph.nodes[node].get(details.get("attribute")) if node else None

    if query_type == "find_objects_by_attribute_value":
        attribute = details.get("attribute")
        wanted = normalise_text(details.get("value"))
        matches = [
            data.get("title")
            for _, data in graph.nodes(data=True)
            if data.get("type") == "object"
            and normalise_text(data.get(attribute)) == wanted
            and data.get("title")
        ]
        return sorted(matches) or None

    if query_type == "find_related":
        node = _find_object(graph, details.get("object_title", ""))
        if not node:
            return None
        matches = [
            _display_name(graph, neighbor)
            for neighbor in graph.neighbors(node)
            if _edge_has_relationship(graph, node, neighbor, details.get("relationship", ""))
        ]
        if not matches:
            return None
        return matches[0] if len(matches) == 1 else sorted(matches)

    if query_type == "find_attribute_of_related_object":
        node = _find_object(graph, details.get("object_title", ""))
        if not node:
            return None
        for neighbor in graph.neighbors(node):
            if _edge_has_relationship(graph, node, neighbor, details.get("relationship", "")):
                target_attribute = details.get("target_attribute")
                if target_attribute == "title":
                    return _display_name(graph, neighbor)
                return graph.nodes[neighbor].get(target_attribute)
        return None

    raise ValueError(f"Unsupported query_type: {query_type}")


def evaluate(graph: nx.Graph, qa_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    mismatches: list[dict[str, Any]] = []
    total_latency = 0.0

    for pair in qa_pairs:
        started = time.perf_counter()
        actual = query_graph(graph, pair["query_type"], pair["query_details"])
        total_latency += time.perf_counter() - started
        if answers_equal(pair.get("answer"), actual):
            correct += 1
        else:
            mismatches.append(
                {
                    "question": pair.get("question"),
                    "expected_answer": pair.get("answer"),
                    "actual_answer": actual,
                }
            )

    total = len(qa_pairs)
    accuracy = correct / total if total else 0.0
    return {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "average_latency_seconds": total_latency / total if total else 0.0,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    graph = nx.read_gexf(args.graph)
    with args.qa.open(encoding="utf-8") as stream:
        qa_pairs = json.load(stream)
    results = evaluate(graph, qa_pairs)
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
