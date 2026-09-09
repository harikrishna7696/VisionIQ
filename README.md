# VisionIQ

> **🚧 Under development** — this project is a work in progress. Some components described below (storage, RAG query layer, deployment) are still being built. See the [Roadmap](#roadmap) for current status.

Natural-language search over video footage. Instead of scrubbing through hours of camera streams, type a question — *"show me every instance of an unauthorized vehicle near gate 12 after 10pm"* — and get back the matching clips with timestamps.

Combines a production-style detection pipeline with a modern VLM + RAG layer: computer vision for events, vision-language models for understanding, and an LLM for conversational retrieval.

## How it works

1. **Detection** — YOLOv7 runs on video streams and writes structured events (timestamp, camera ID, class, bounding box, track ID) to sidecar JSONL files.
2. **Captioning & embeddings** — Each detected event is captioned with Moondream and embedded with CLIP, making it semantically searchable.
3. **Storage** — Embeddings go into a vector store (ChromaDB/Qdrant); structured fields go into a SQL table for exact filtering.
4. **Retrieval** — A hybrid retriever combines vector similarity search with SQL filters and text-to-SQL queries.
5. **Answering** — An LLM (via LangChain/LlamaIndex) synthesizes a natural-language answer from retrieved events, with clip timestamps and citations.
6. **Interface** — A FastAPI backend serves a Streamlit demo UI with clip playback.

## Roadmap
- [x] Detection pipeline & event schema
- [x] VLM captioning & embeddings
- [ ] Vector + SQL storage layer
- [ ] RAG query layer
- [ ] Text-to-SQL integration
- [ ] Demo interface
- [ ] Deployment