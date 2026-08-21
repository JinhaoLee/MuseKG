"""Exact-match and optional local-LLM evaluation for MuseKG QA pairs."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol

import networkx as nx


@dataclass(frozen=True)
class JudgeDecision:
    """A normalized answer-correctness decision."""

    correct: bool
    explanation: str | None = None


class AnswerJudge(Protocol):
    """Interface shared by exact-match and model-based judges."""

    name: str

    def judge(self, question: Any, expected: Any, actual: Any) -> JudgeDecision: ...


class ExactMatchJudge:
    """Compare normalized answer sets without loading a model."""

    name = "exact"

    def judge(self, question: Any, expected: Any, actual: Any) -> JudgeDecision:
        del question
        return JudgeDecision(correct=answers_equal(expected, actual))


def _render_answer(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_judge_response(response: str) -> JudgeDecision:
    """Extract the first valid judge decision JSON object from model output."""

    decoder = json.JSONDecoder()
    for offset, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[offset:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        correctness = str(payload.get("correctness", "")).strip().casefold()
        if correctness not in {"correct", "incorrect"}:
            continue
        explanation = payload.get("explanation")
        return JudgeDecision(
            correct=correctness == "correct",
            explanation=str(explanation).strip() if explanation is not None else None,
        )
    raise ValueError("LLM judge did not return valid correctness JSON")


Generator = Callable[..., list[dict[str, Any]]]


class LLMJudge:
    """Judge answers with an explicitly selected local Hugging Face model."""

    name = "llm"

    def __init__(
        self,
        model_name: str,
        prompt_template: str | None = None,
        max_new_tokens: int = 128,
        generator: Generator | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name is required for LLM judging")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.model_name = model_name
        self.prompt_template = prompt_template or load_default_judge_prompt()
        self.max_new_tokens = max_new_tokens
        self._generator = generator

    def _load_generator(self) -> Generator:
        import torch
        from transformers import pipeline

        return pipeline(
            "text-generation",
            model=self.model_name,
            device=0 if torch.cuda.is_available() else -1,
            trust_remote_code=False,
        )

    def judge(self, question: Any, expected: Any, actual: Any) -> JudgeDecision:
        if self._generator is None:
            self._generator = self._load_generator()
        prompt = self.prompt_template.format(
            question=_render_answer(question),
            expected_answer=_render_answer(expected),
            kg_answer=_render_answer(actual),
        )
        outputs = self._generator(
            prompt,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        if (
            not outputs
            or not isinstance(outputs[0], dict)
            or not isinstance(outputs[0].get("generated_text"), str)
        ):
            raise ValueError("LLM judge returned no generated text")
        generated = outputs[0]["generated_text"]
        completion = generated.removeprefix(prompt)
        return parse_judge_response(completion)


def load_default_judge_prompt() -> str:
    """Load the packaged copy of the public evaluation prompt."""

    return files("musekg.prompts").joinpath("eval_prompt.txt").read_text(encoding="utf-8")


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


def evaluate(
    graph: nx.Graph,
    qa_pairs: list[dict[str, Any]],
    judge: AnswerJudge | None = None,
) -> dict[str, Any]:
    selected_judge = judge or ExactMatchJudge()
    correct = 0
    mismatches: list[dict[str, Any]] = []
    total_query_latency = 0.0
    total_judge_latency = 0.0

    for pair in qa_pairs:
        started = time.perf_counter()
        actual = query_graph(graph, pair["query_type"], pair["query_details"])
        total_query_latency += time.perf_counter() - started

        started = time.perf_counter()
        decision = selected_judge.judge(pair.get("question"), pair.get("answer"), actual)
        total_judge_latency += time.perf_counter() - started
        if decision.correct:
            correct += 1
        else:
            mismatch = {
                "question": pair.get("question"),
                "expected_answer": pair.get("answer"),
                "actual_answer": actual,
            }
            if decision.explanation:
                mismatch["judge_explanation"] = decision.explanation
            mismatches.append(mismatch)

    total = len(qa_pairs)
    accuracy = correct / total if total else 0.0
    return {
        "judge": selected_judge.name,
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "average_latency_seconds": total_query_latency / total if total else 0.0,
        "average_judge_latency_seconds": total_judge_latency / total if total else 0.0,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--judge",
        choices=("exact", "llm"),
        default="exact",
        help="Answer judge to use (default: exact)",
    )
    parser.add_argument("--judge-model", help="Local path or Hugging Face model ID for LLM judging")
    parser.add_argument("--judge-prompt", type=Path, help="Optional replacement judge prompt")
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.judge == "llm" and not args.judge_model:
        parser.error("--judge-model is required when --judge llm")
    if args.judge_max_new_tokens < 1:
        parser.error("--judge-max-new-tokens must be positive")

    graph = nx.read_gexf(args.graph)
    with args.qa.open(encoding="utf-8") as stream:
        qa_pairs = json.load(stream)
    judge: AnswerJudge
    if args.judge == "llm":
        prompt_template = (
            args.judge_prompt.read_text(encoding="utf-8") if args.judge_prompt else None
        )
        judge = LLMJudge(
            model_name=args.judge_model,
            prompt_template=prompt_template,
            max_new_tokens=args.judge_max_new_tokens,
        )
    else:
        judge = ExactMatchJudge()
    results = evaluate(graph, qa_pairs, judge=judge)
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
