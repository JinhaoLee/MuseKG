import json
from pathlib import Path

import networkx as nx

from musekg.demo_system import MuseKGSystem
from musekg.evaluation import evaluate
from musekg.graph import build_knowledge_graph

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "fictional_museum"


def test_fictional_collection_builds_and_evaluates(tmp_path: Path) -> None:
    graph = build_knowledge_graph(
        EXAMPLE_ROOT / "objects.json",
        EXAMPLE_ROOT / "entities.json",
        EXAMPLE_ROOT / "ocr.json",
    )

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 5
    assert graph.nodes["object:SYN-003"]["ocr_text"].startswith("SAMPLE")

    graph_path = tmp_path / "fictional.gexf"
    nx.write_gexf(graph, graph_path)
    restored = nx.read_gexf(graph_path)
    qa_pairs = json.loads((EXAMPLE_ROOT / "qa_pairs.json").read_text())
    results = evaluate(restored, qa_pairs)

    assert results["judge"] == "exact"
    assert results["correct_answers"] == 3
    assert results["accuracy"] == 1.0
    assert results["mismatches"] == []


def test_demo_uses_local_deterministic_answer(tmp_path: Path) -> None:
    graph = build_knowledge_graph(EXAMPLE_ROOT / "objects.json")
    graph_path = tmp_path / "fictional.gexf"
    nx.write_gexf(graph, graph_path)

    system = MuseKGSystem(str(graph_path))
    system.load_resources()
    result = system.generate_answer("What material is the Demonstration Brass Telescope made from?")

    assert result["answer"] == "Brass and glass"
    assert result["subgraph_nodes"]

    producer = system.generate_answer("Who produced the Demonstration Brass Telescope?")
    assert producer["answer"] == "Example Scientific Works"


def test_unknown_object_does_not_expose_graph_context(tmp_path: Path) -> None:
    graph = build_knowledge_graph(EXAMPLE_ROOT / "objects.json")
    graph_path = tmp_path / "fictional.gexf"
    nx.write_gexf(graph, graph_path)

    system = MuseKGSystem(str(graph_path))
    system.load_resources()
    result = system.generate_answer("Tell me about an object that is not present")

    assert result["subgraph_nodes"] == []
    assert result["context"] == "No matching object was found."
