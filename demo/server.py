"""FastAPI server for a locally hosted MuseKG demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from musekg.demo_system import MuseKGSystem

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(os.getenv("MUSEKG_STATIC_PATH", REPOSITORY_ROOT / "demo" / "static"))
GRAPH_PATH = os.getenv(
    "MUSEKG_GRAPH_PATH",
    str(REPOSITORY_ROOT / "output" / "fictional_knowledge_graph.gexf"),
)
MODEL_NAME = os.getenv("MUSEKG_MODEL") or None

system = MuseKGSystem(graph_file=GRAPH_PATH, model_name=MODEL_NAME)


@asynccontextmanager
async def lifespan(_: FastAPI):
    system.load_resources()
    yield


app = FastAPI(title="MuseKG demo", lifespan=lifespan)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "MUSEKG_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class ChatResponse(BaseModel):
    answer: str
    context: str
    entity: str
    graph_data: dict


def _edge_label(subgraph, source, target, key=None) -> str:
    if subgraph.is_multigraph():
        return str(
            (subgraph.get_edge_data(source, target, key) or {}).get("relationship", "related_to")
        )
    return str((subgraph.get_edge_data(source, target) or {}).get("relationship", "related_to"))


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = system.generate_answer(request.query)
    selected = result["subgraph_nodes"]
    nodes: list[dict] = []
    edges: list[dict] = []

    if selected and system.kg is not None:
        subgraph = system.kg.subgraph(selected)
        for node, attributes in subgraph.nodes(data=True):
            rendered = dict(attributes)
            rendered["id"] = str(node)
            rendered["label"] = rendered.get("title", str(node))
            rendered.setdefault("type", "node")
            nodes.append(rendered)
        if subgraph.is_multigraph():
            for source, target, key in subgraph.edges(keys=True):
                edges.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "label": _edge_label(subgraph, source, target, key),
                    }
                )
        else:
            for source, target in subgraph.edges():
                edges.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "label": _edge_label(subgraph, source, target),
                    }
                )

    return ChatResponse(
        answer=result["answer"],
        context=result["context"],
        entity=result["entity_extracted"],
        graph_data={"nodes": nodes, "edges": edges},
    )


@app.get("/health")
def health() -> dict:
    graph_loaded = system.kg is not None and system.kg.number_of_nodes() > 0
    return {
        "status": "ok" if graph_loaded else "degraded",
        "ready": graph_loaded,
        "graph_loaded": graph_loaded,
        "llm_enabled": system.llm_pipeline is not None,
    }


app.mount("/", StaticFiles(directory=STATIC_ROOT, html=True), name="static")
