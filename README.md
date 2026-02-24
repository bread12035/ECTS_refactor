# ECTS Refactor — Langflow → LangGraph

A LangGraph-based Python application that replicates an existing Langflow workflow as a REST API. The workflow processes an earnings-call transcript and associated metadata through four parallel branches, merges the results, and returns a structured AI summary.

---

## Project Structure

```
ECTS_refactor/
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
├── state.py              # GraphState TypedDict (shared state schema)
├── prompts.py            # Prompt configuration (fill in before running)
├── llm.py                # _get_llm() utility — shared LLM factory
├── nodes.py              # All LangGraph node functions
├── graph.py              # StateGraph compilation and wiring
└── app.py                # FastAPI application entry point
```

---

## Setup

**1. Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key for the LLM provider |
| `OPENAI_API_BASE` | Base URL of the LLM endpoint (e.g. `https://api.openai.com/v1`) |

**3. Fill in prompts**

Open `prompts.py` and populate the `system` and `user` strings for each node before running the application. All strings are intentionally left empty as placeholders.

**4. Run the server**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API

### `POST /run`

**Request body**

```json
{
    "report_template": "<string>",
    "transcript":      "<string>",
    "segment_data":    "<string>",
    "segment_items":   "<string>"
}
```

**Response body**

```json
{
    "outputs": [
        {
            "outputs": [
                {
                    "results": {
                        "message": {
                            "text": "<Final AI Summary String>"
                        }
                    }
                }
            ]
        }
    ]
}
```

---

## Workflow Overview

```
START
  └─► start_point
        ├─► [Branch 1] transcript_fa_extraction → fa_highlights ─────────────────────┐
        ├─► [Branch 2] transcript_guidance_extraction → guid_validation ──────────────┤
        ├─► [Branch 3] segment_extraction → context_retrieval ──┬─► integrator        │
        │                                                       │     └─► key_messages ┤
        │                                                       └─► summarizer         │
        │                                                             └─► briefing_key_messages ─┤
        └─► [Branch 4] transcript_QA_extraction → second_QA ───────────────────────────┤
                                                                             output_template
                                                                                   └─► wrapper → END
```

| Branch | Purpose | Key outputs |
|---|---|---|
| 1 – Financial Highlights | Extracts FA-relevant content and generates highlights | `fa_highlights_out` |
| 2 – Guidance | Extracts and validates guidance statements | `guid_validation_out` |
| 3 – Key Message | Segment-aware extraction with parallel integrator/summarizer paths | `key_messages_out`, `briefing_key_messages_out` |
| 4 – QA | Two-pass Q&A extraction and review | `second_QA_out` |

`output_template` waits for all four branches, then generates the final `ai_summary`. `wrapper` formats it into the required JSON envelope.

---

## Design Notes

- **Prompt decoupling** — all system/user prompts live in `prompts.py`. Node logic contains no hardcoded prompt text.
- **Shared LLM factory** — every LLM node calls `_get_llm()` from `llm.py`. To swap models or adjust parameters, edit that one function.
- **Structured output nodes** — `segment_extraction` and `context_retrieval` return `dict` values (`{"content": ...}`). The schema can be tightened with `.with_structured_output()` once a response schema is defined.
- **No LLM in `start_point` / `wrapper`** — these nodes perform pure data transformation only.
