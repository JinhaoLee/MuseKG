from pathlib import Path

import networkx as nx
import pytest

from musekg.evaluation import (
    LLMJudge,
    evaluate,
    load_default_judge_prompt,
    parse_judge_response,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_judge_prompt_matches_public_prompt() -> None:
    public_prompt = (REPOSITORY_ROOT / "prompts" / "eval_prompt.txt").read_text(encoding="utf-8")
    assert load_default_judge_prompt() == public_prompt


def test_parse_judge_response_accepts_fenced_json() -> None:
    decision = parse_judge_response(
        '```json\n{"correctness":"incorrect","explanation":"Different values."}\n```'
    )

    assert decision.correct is False
    assert decision.explanation == "Different values."


def test_llm_judge_formats_prompt_and_parses_completion() -> None:
    prompts: list[str] = []

    def generator(prompt: str, **_: object) -> list[dict[str, str]]:
        prompts.append(prompt)
        return [
            {
                "generated_text": prompt
                + '\n{"correctness":"correct","explanation":"Semantically equivalent."}'
            }
        ]

    judge = LLMJudge(model_name="fictional-local-model", generator=generator)
    decision = judge.judge("What is it made from?", "brass and glass", "glass and brass")

    assert decision.correct is True
    assert decision.explanation == "Semantically equivalent."
    assert "What is it made from?" in prompts[0]
    assert "brass and glass" in prompts[0]
    assert "glass and brass" in prompts[0]


def test_evaluate_can_use_llm_judge_for_semantic_equivalence() -> None:
    graph = nx.MultiDiGraph()
    graph.add_node("object:1", type="object", title="Fictional Object", material_desc="brass")
    qa_pairs = [
        {
            "question": "What is the fictional object made from?",
            "answer": "made from brass",
            "query_type": "find_attribute",
            "query_details": {
                "object_title": "Fictional Object",
                "attribute": "material_desc",
            },
        }
    ]

    def generator(prompt: str, **_: object) -> list[dict[str, str]]:
        completion = '\n{"correctness":"correct","explanation":"Same fact."}'
        return [{"generated_text": prompt + completion}]

    exact_results = evaluate(graph, qa_pairs)
    llm_results = evaluate(
        graph,
        qa_pairs,
        judge=LLMJudge(model_name="fictional-local-model", generator=generator),
    )

    assert exact_results["correct_answers"] == 0
    assert llm_results["judge"] == "llm"
    assert llm_results["correct_answers"] == 1


def test_parse_judge_response_rejects_missing_decision() -> None:
    with pytest.raises(ValueError, match="valid correctness JSON"):
        parse_judge_response("The answers appear similar.")
