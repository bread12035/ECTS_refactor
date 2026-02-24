"""
FastAPI application entry point.

Run locally with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

POST /run
---------
Request body (JSON):
    {
        "report_template": "<string>",
        "transcript": "<string>",
        "segment_data": "<string>",
        "segment_items": "<string>"
    }

Response body (JSON):
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
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import workflow

app = FastAPI(
    title="ECTS LangGraph API",
    description="Processes transcripts and segment data through a multi-branch LangGraph workflow.",
    version="1.0.0",
)


class RequestPayload(BaseModel):
    report_template: str
    transcript: str
    segment_data: str
    segment_items: str


@app.post("/run")
async def run_workflow(payload: RequestPayload) -> dict:
    """
    Execute the full LangGraph workflow and return the wrapped AI summary.
    """
    try:
        initial_state = {
            "report_template": payload.report_template,
            "transcript": payload.transcript,
            "segment_data": payload.segment_data,
            "segment_items": payload.segment_items,
        }
        result = await workflow.ainvoke(initial_state)
        return result["final_response"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
