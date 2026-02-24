from typing import TypedDict


class GraphState(TypedDict, total=False):
    # Inputs
    report_template: str
    transcript: str
    segment_data: str
    segment_items: str

    # Branch 1 Intermediate
    transcript_fa_extracted: str
    fa_highlights_out: str

    # Branch 2 Intermediate
    transcript_guidance_extracted: str
    guid_validation_out: str

    # Branch 3 Intermediate
    segment_extraction_out: dict  # Structured Output
    context_retrieval_out: dict   # Structured Output
    integrator_out: str
    key_messages_out: str
    summarizer_out: str
    briefing_key_messages_out: str

    # Branch 4 Intermediate
    transcript_QA_extracted: str
    second_QA_out: str

    # Merged Output & Final Wrap
    ai_summary: str
    final_response: dict
