# MuseKG

MuseKG is an interactive knowledge-graph system for constructing and querying museum
collections. It combines structured museum records, optional OCR and extracted entities,
deterministic graph evaluation, research prompt templates, and a local web demo.

This repository accompanies **“MuseKG: An Interactive Knowledge Graph Over Museum
Collections”**, published at SIGIR 2026 ([DOI: 10.1145/3805712.3808366](https://doi.org/10.1145/3805712.3808366)).

## Privacy and release scope

The museum collection used in the paper is private and is **not included**. The repository
contains a wholly fictional three-object collection so that every public component can be
run end to end. Users can supply their own records using the documented input contract.
See [the privacy boundary](docs/privacy.md) before using sensitive data.

## Included

- Streaming graph construction from museum JSON records
- Optional OCR and extracted-entity enrichment
- Deterministic evaluation for structured QA pairs
- Prompt templates used by the research pipeline
- A FastAPI and NetworkX demo with an optional local Hugging Face model
- A fictional dataset, tests, and publication-safety checks

## Installation

MuseKG requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install the lightweight web-demo dependencies with:

```bash
python -m pip install -e ".[demo]"
```

Only if you want local Hugging Face generation, install the larger model dependencies:

```bash
python -m pip install -e ".[model]"
```

## Build the fictional graph

```bash
musekg-build \
  --objects examples/fictional_museum/objects.json \
  --entities examples/fictional_museum/entities.json \
  --ocr examples/fictional_museum/ocr.json \
  --output output/fictional_knowledge_graph.gexf
```

The inputs stay on the local machine. To use a different collection, follow
[the data-format guide](docs/data-format.md) and point these arguments to local files.

## Evaluate

```bash
musekg-evaluate \
  --graph output/fictional_knowledge_graph.gexf \
  --qa examples/fictional_museum/qa_pairs.json \
  --output output/fictional_evaluation.json
```

The fictional benchmark should report 3/3 correct answers. This validates the public
pipeline; it does not reproduce the paper's private-data results.

## Run the local demo

```bash
MUSEKG_GRAPH_PATH=output/fictional_knowledge_graph.gexf \
  uvicorn demo.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. By default the demo uses deterministic answers and needs no
GPU. To enable an optional local model, install the `model` extra and set `MUSEKG_MODEL`
to a compatible Hugging Face model identifier. MuseKG does not enable remote model code.
The graph visualisation loads a version-pinned Cytoscape.js bundle from jsDelivr, so that
part of the interface needs an internet connection.

Do not expose a private graph through a public demo: responses, context, and visualised
neighbourhoods can reveal the underlying records.

## Prompts and generated code

The `prompts/` directory records research prompt templates. Some templates ask a model to
produce NetworkX code for experimental comparison. This public release deliberately does
not execute model-generated Python. Treat generated text as untrusted input.

## Development

```bash
pytest
ruff check .
ruff format --check .
python scripts/check_publication.py
```

## Citation

GitHub can generate a citation from [`CITATION.cff`](CITATION.cff). Please cite the paper
when using MuseKG in academic work.

## License

No open-source license has been selected yet. Until a `LICENSE` file is added, copyright
remains with the authors and normal copyright restrictions apply.
