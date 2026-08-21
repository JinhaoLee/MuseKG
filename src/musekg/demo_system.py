"""Privacy-conscious local query system used by the MuseKG demo."""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

import networkx as nx

from .evaluation import _display_name, _edge_has_relationship, normalise_text

LOGGER = logging.getLogger("musekg.demo")


class MuseKGSystem:
    def __init__(self, graph_file: str, model_name: str | None = None):
        self.graph_file = graph_file
        self.model_name = model_name
        self.kg: nx.Graph | None = None
        self.llm_pipeline: Any = None
        self._titles: dict[str, str] = {}

    def load_resources(self) -> None:
        self.kg = nx.read_gexf(self.graph_file)
        self._titles = {
            normalise_text(data.get("title")) or "": node
            for node, data in self.kg.nodes(data=True)
            if data.get("title")
        }
        if self.model_name:
            import torch
            from transformers import pipeline

            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=128,
                trust_remote_code=False,
            )

    def _extract_entity(self, query: str) -> str:
        lowered = normalise_text(query) or ""
        contained = [title for title in self._titles if title and title in lowered]
        if contained:
            return max(contained, key=len)
        cleaned = re.sub(
            r"^(what|who|where|when|how|which|show me|tell me|describe)\s+",
            "",
            lowered,
        ).rstrip("?.,!")
        matches = difflib.get_close_matches(cleaned, self._titles, n=1, cutoff=0.55)
        return matches[0] if matches else cleaned

    def _context(self, node: str) -> tuple[str, list[str]]:
        assert self.kg is not None
        data = self.kg.nodes[node]
        lines = [f"The object titled '{data.get('title', node)}' has these properties:"]
        for key, value in data.items():
            if key not in {"id", "label", "source_id", "title", "type"}:
                lines.append(f"- {key}: {value}")
        nodes = [node]
        for neighbor in self.kg.neighbors(node):
            nodes.append(neighbor)
            edge_data = self.kg.get_edge_data(node, neighbor) or {}
            if self.kg.is_multigraph():
                relationships = [
                    item.get("relationship", "related_to") for item in edge_data.values()
                ]
            else:
                relationships = [edge_data.get("relationship", "related_to")]
            for relationship in relationships:
                lines.append(f"- {relationship}: {_display_name(self.kg, neighbor)}")
        return "\n".join(lines), nodes

    def _deterministic_answer(self, query: str, node: str) -> str:
        assert self.kg is not None
        lowered = query.casefold()
        data = self.kg.nodes[node]
        attributes = {
            "material": "material_desc",
            "accession": "accession_no",
            "museum": "museum",
            "category": "history_cat",
            "description": "description",
        }
        for word, attribute in attributes.items():
            if word in lowered and data.get(attribute):
                return str(data[attribute])
        if any(word in lowered for word in ("who produced", "producer", "creator")):
            for neighbor in self.kg.neighbors(node):
                for relationship in (
                    "has_primary_producer",
                    "has_primary_organisation",
                ):
                    if _edge_has_relationship(self.kg, node, neighbor, relationship):
                        return _display_name(self.kg, neighbor)
        available = [
            f"{key}: {data[key]}"
            for key in ("description", "material_desc", "production_date", "museum")
            if data.get(key)
        ]
        return "; ".join(available) or "No descriptive attributes are available."

    def generate_answer(self, query: str) -> dict[str, Any]:
        if self.kg is None:
            raise RuntimeError("MuseKG resources have not been loaded")
        entity = self._extract_entity(query)
        node = self._titles.get(entity)
        if not node:
            return {
                "query": query,
                "entity_extracted": entity,
                "context": "No matching object was found.",
                "answer": "I could not find a matching object in this collection.",
                "subgraph_nodes": [],
            }
        context, nodes = self._context(node)
        if self.llm_pipeline:
            prompt = (
                "Answer the question using only the supplied context.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            generated = self.llm_pipeline(prompt, do_sample=False)[0]["generated_text"]
            answer = generated.removeprefix(prompt).strip().splitlines()[0]
        else:
            answer = self._deterministic_answer(query, node)
        return {
            "query": query,
            "entity_extracted": entity,
            "context": context,
            "answer": answer,
            "subgraph_nodes": nodes,
        }
