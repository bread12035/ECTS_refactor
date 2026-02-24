"""
LangGraph workflow definition.

Graph execution flow
--------------------
START
  └─► json_parser
        ├─► transcript_fa_extraction ─► fa_highlights ─────────────────────────────────┐
        ├─► transcript_guidance_extraction ─► guid_validation ─────────────────────────┤
        ├─► segment_extraction ─► context_retrieval ─┬─► integrator ─► key_messages ───┤
        │                                            └─► summarizer ─► briefing_key_messages ─┤
        └─► transcript_QA_extraction ─► second_QA ──────────────────────────────────────┤
                                                                               output_template
                                                                                     └─► wrapper ─► END
"""

from langgraph.graph import END, START, StateGraph

from nodes import (
    briefing_key_messages_node,
    context_retrieval_node,
    fa_highlights_node,
    guid_validation_node,
    integrator_node,
    json_parser_node,
    key_messages_node,
    output_template_node,
    second_QA_node,
    segment_extraction_node,
    summarizer_node,
    transcript_QA_extraction_node,
    transcript_fa_extraction_node,
    transcript_guidance_extraction_node,
    wrapper_node,
)
from state import GraphState


def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    # ------------------------------------------------------------------
    # Register all nodes
    # ------------------------------------------------------------------
    builder.add_node("json_parser", json_parser_node)

    # Branch 1
    builder.add_node("transcript_fa_extraction", transcript_fa_extraction_node)
    builder.add_node("fa_highlights", fa_highlights_node)

    # Branch 2
    builder.add_node("transcript_guidance_extraction", transcript_guidance_extraction_node)
    builder.add_node("guid_validation", guid_validation_node)

    # Branch 3
    builder.add_node("segment_extraction", segment_extraction_node)
    builder.add_node("context_retrieval", context_retrieval_node)
    builder.add_node("integrator", integrator_node)
    builder.add_node("key_messages", key_messages_node)
    builder.add_node("summarizer", summarizer_node)
    builder.add_node("briefing_key_messages", briefing_key_messages_node)

    # Branch 4
    builder.add_node("transcript_QA_extraction", transcript_QA_extraction_node)
    builder.add_node("second_QA", second_QA_node)

    # Aggregation & wrapping
    builder.add_node("output_template", output_template_node)
    builder.add_node("wrapper", wrapper_node)

    # ------------------------------------------------------------------
    # Define edges
    # ------------------------------------------------------------------

    # Entry point
    builder.add_edge(START, "json_parser")

    # Fan-out: json_parser → all four branch heads (executed in parallel)
    builder.add_edge("json_parser", "transcript_fa_extraction")
    builder.add_edge("json_parser", "transcript_guidance_extraction")
    builder.add_edge("json_parser", "segment_extraction")
    builder.add_edge("json_parser", "transcript_QA_extraction")

    # Branch 1: Financial Highlights
    builder.add_edge("transcript_fa_extraction", "fa_highlights")
    builder.add_edge("fa_highlights", "output_template")

    # Branch 2: Guidance
    builder.add_edge("transcript_guidance_extraction", "guid_validation")
    builder.add_edge("guid_validation", "output_template")

    # Branch 3: Key Message
    #   segment_extraction → context_retrieval
    #   context_retrieval fans out to integrator and summarizer (parallel)
    builder.add_edge("segment_extraction", "context_retrieval")
    builder.add_edge("context_retrieval", "integrator")
    builder.add_edge("context_retrieval", "summarizer")
    builder.add_edge("integrator", "key_messages")
    builder.add_edge("key_messages", "output_template")
    builder.add_edge("summarizer", "briefing_key_messages")
    builder.add_edge("briefing_key_messages", "output_template")

    # Branch 4: QA
    builder.add_edge("transcript_QA_extraction", "second_QA")
    builder.add_edge("second_QA", "output_template")

    # Fan-in at output_template; then wrapper; then END
    builder.add_edge("output_template", "wrapper")
    builder.add_edge("wrapper", END)

    return builder.compile()


# Module-level compiled workflow, ready for import by app.py
workflow = build_graph()
