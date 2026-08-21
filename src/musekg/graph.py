"""Build a MuseKG graph from user-supplied museum records."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import ijson
import networkx as nx

RELATIONSHIP_MAPPING = {
    "object_prod_pri_person": "has_primary_producer",
    "object_au_related": "has_related_object",
    "object_prod_sec_person": "has_secondary_producer",
    "object_prod_pri_org": "has_primary_organisation",
    "object_prod_sec_org": "has_secondary_organisation",
}

OBJECT_ATTRIBUTES = {
    "accession_no",
    "collection",
    "credit_line",
    "description",
    "history_cat",
    "material_desc",
    "measurements",
    "museum",
    "name",
    "object_type",
    "production_date",
}


def _items(path: Path, prefix: str) -> Iterable[dict[str, Any]]:
    with path.open("rb") as stream:
        yield from ijson.items(stream, prefix)


def _normalise_identifier(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _record_node_id(record_type: Any, record_id: Any) -> str:
    kind = _normalise_identifier(record_type or "record")
    identifier = str(record_id).strip()
    if kind == "object":
        return f"object:{identifier}"
    return f"record:{kind}:{identifier}"


def _object_node_id(object_id: Any) -> str:
    return _record_node_id("object", object_id)


def _add_object_fields(graph: nx.MultiDiGraph, node_id: str, obj: dict[str, Any]) -> None:
    for field_set in obj.get("opacObjectFieldSets") or []:
        identifier = field_set.get("identifier")
        fields = field_set.get("opacObjectFields") or []
        if identifier not in OBJECT_ATTRIBUTES or not fields:
            continue
        value = fields[0].get("value")
        if value in (None, ""):
            continue
        graph.nodes[node_id]["title" if identifier == "name" else identifier] = value


def _add_relationships(
    graph: nx.MultiDiGraph,
    object_node: str,
    obj: dict[str, Any],
    relationship_types: set[str],
) -> None:
    relationships = (obj.get("relationshipsCollection") or {}).get("relationships") or []
    for relationship in relationships:
        raw_type = relationship.get("relationshipId")
        relation = RELATIONSHIP_MAPPING.get(raw_type, raw_type)
        if not relation:
            continue
        for related in relationship.get("relatedRecords") or []:
            related_id = related.get("relatedRecordId")
            title = related.get("title")
            if related_id in (None, "") or not title:
                continue
            related_type = related.get("relatedRecordType") or "record"
            related_node = _record_node_id(related_type, related_id)
            graph.add_node(related_node, type=str(related_type).casefold(), title=title)
            graph.add_edge(object_node, related_node, relationship=relation)
            relationship_types.add(relation)


def _add_image_labels(
    graph: nx.MultiDiGraph,
    object_node: str,
    obj: dict[str, Any],
    relationship_types: set[str],
) -> None:
    images = (obj.get("imagesCollection") or {}).get("images") or []
    for image in images:
        for label in image.get("imageLabels") or []:
            label_name = label.get("imageLabel")
            if not label_name:
                continue
            label_node = f"label:{_normalise_identifier(label_name)}"
            graph.add_node(label_node, type="image_label", title=label_name)
            graph.add_edge(object_node, label_node, relationship="has_label")
            relationship_types.add("has_label")


def _add_entities(graph: nx.MultiDiGraph, entities_path: Path) -> None:
    for entity_info in _items(entities_path, "results.item"):
        object_node = _object_node_id(entity_info.get("object_id"))
        if object_node not in graph:
            continue
        entities = entity_info.get("entities") or {}
        if not isinstance(entities, dict):
            continue
        for entity_type, values in entities.items():
            for entity in values if isinstance(values, list) else []:
                name = entity.get("name") if isinstance(entity, dict) else None
                if not name:
                    continue
                entity_node = (
                    f"entity:{_normalise_identifier(entity_type)}:{_normalise_identifier(name)}"
                )
                graph.add_node(entity_node, type=str(entity_type), title=name)
                graph.add_edge(object_node, entity_node, relationship="has_entity")


def _add_ocr(graph: nx.MultiDiGraph, ocr_path: Path) -> None:
    for result in _items(ocr_path, "results.item"):
        image_name = result.get("image_name")
        if not image_name:
            continue
        object_node = _object_node_id(Path(str(image_name)).stem)
        if object_node in graph and result.get("extracted_text") is not None:
            graph.nodes[object_node]["ocr_text"] = result["extracted_text"]


def build_knowledge_graph(
    objects_path: str | Path,
    entities_path: str | Path | None = None,
    ocr_path: str | Path | None = None,
) -> nx.MultiDiGraph:
    """Build a graph without retaining or transmitting the input files."""

    graph = nx.MultiDiGraph()
    relationship_types: set[str] = set()

    for obj in _items(Path(objects_path), "item"):
        object_id = obj.get("opacObjectId")
        if object_id in (None, ""):
            continue
        object_node = _object_node_id(object_id)
        graph.add_node(object_node, type="object", source_id=str(object_id))
        _add_object_fields(graph, object_node, obj)
        _add_relationships(graph, object_node, obj, relationship_types)
        _add_image_labels(graph, object_node, obj, relationship_types)

    if entities_path:
        _add_entities(graph, Path(entities_path))
        relationship_types.add("has_entity")
    if ocr_path:
        _add_ocr(graph, Path(ocr_path))

    graph.graph["relationship_types"] = ",".join(sorted(relationship_types))
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", required=True, type=Path, help="Museum object JSON array")
    parser.add_argument("--entities", type=Path, help="Optional extracted-entity JSON")
    parser.add_argument("--ocr", type=Path, help="Optional OCR JSON")
    parser.add_argument("--output", required=True, type=Path, help="Output GEXF path")
    args = parser.parse_args()

    graph = build_knowledge_graph(args.objects, args.entities, args.ocr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, args.output)
    print(
        f"Wrote {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
