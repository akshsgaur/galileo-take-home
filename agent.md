# Research Agent Overview

## Key Features
- Six-stage LangGraph workflow: **Plan → Search → Curate → Analyze → Synthesize → Validate**
- Galileo observability on every span (per-step metrics, Luna scorers, trace links)
- Tavily-powered search with query expansion + multi-source deduplication
- Source curation heuristics (domain trust, confidence scoring, reason strings)
- LLM-as-judge evaluations (plan quality, search relevance, analysis completeness, answer quality, groundedness)
- FastAPI wrapper returns `trace_id`, `trace_url`, curated sources, and step metrics
- Next.js UI mirrors the pipeline (live metrics, source metadata, Galileo deep links)

## Main Files

| File | Purpose |
|------|---------|
| `agent.py` | LangGraph workflow + Galileo instrumentation |
| `tools.py` | Tavily search helper |
| `evaluators.py` | LLM-based evaluators + groundedness judge |
| `api_server.py` | FastAPI service for UI/API consumers |
| `research-agent-ui/` | Next.js frontend (answer/sources/metrics tabs) |

## Workflow Notes
- `agent.py` state includes `search_results`, `curated_sources`, and metrics.
- Curate step promotes high-confidence sources (max 8) with domain tags.
- Validate step appends warnings if groundedness <= 6.
- LangGraph run is executed with `GalileoCallback` so traces are captured automatically.

## Setup Checklist
1. `cp research-agent/.env.example research-agent/.env`
2. Populate `GALILEO_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`
3. `pip install -r research-agent/requirements.txt`
4. `python research-agent/verify_setup.py`

## Useful Commands
- Run single question: `python research-agent/agent.py`
- FastAPI server: `python research-agent/api_server.py`
- Evaluation suite: `python research-agent/evaluate.py`
- Frontend (from `research-agent-ui/`): `npm install && npm run dev`

## Observability Tips
- Trace link: returned as `trace_url` from API; front end surfaces "View in Galileo" button.
- LangGraph runs include `GalileoCallback`, so nodes/metrics appear automatically in the console.
- Log stream metrics: context adherence + hallucination (Luna) available under project/log stream configured at runtime.
- Curate metrics (avg confidence, count) visible in Galileo metadata + UI metrics tab.
