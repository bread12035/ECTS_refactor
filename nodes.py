"""
LangGraph node implementations.

Every node that performs LLM processing must obtain its model via _get_llm().
The two exceptions – json_parser and wrapper – perform pure data transformation
and must NOT call the LLM.

Each node receives the full GraphState and returns a dict containing only the
keys it produces; LangGraph merges the returned dict back into the shared state.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from llm import _get_llm
from prompts import PROMPTS
from state import GraphState


# ---------------------------------------------------------------------------
# Initialization node (no LLM)
# ---------------------------------------------------------------------------

def json_parser_node(state: GraphState) -> dict:
    """
    Entry node. The API layer pre-populates GraphState with the four base
    inputs before graph invocation, so no additional transformation is needed.
    """
    return {}


# ---------------------------------------------------------------------------
# Branch 1: Financial Highlights
# ---------------------------------------------------------------------------

def transcript_fa_extraction_node(state: GraphState) -> dict:
    """Node 1.1 – extract financial-analyst relevant content from the transcript."""
    llm = _get_llm()
    system_prompt = PROMPTS["transcript_fa_extraction"]["system"]
    user_prompt = PROMPTS["transcript_fa_extraction"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"{user_prompt}\n\nTranscript:\n{state['transcript']}"
        ),
    ]
    response = llm.invoke(messages)
    return {"transcript_fa_extracted": response.content}


def fa_highlights_node(state: GraphState) -> dict:
    """Node 1.2 – produce financial-analysis highlights using the report template."""
    llm = _get_llm()
    system_prompt = PROMPTS["fa_highlights"]["system"]
    user_prompt = PROMPTS["fa_highlights"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Report Template:\n{state['report_template']}\n\n"
                f"Extracted FA Transcript:\n{state['transcript_fa_extracted']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"fa_highlights_out": response.content}


# ---------------------------------------------------------------------------
# Branch 2: Guidance
# ---------------------------------------------------------------------------

def transcript_guidance_extraction_node(state: GraphState) -> dict:
    """Node 2.1 – extract guidance-related content from the transcript."""
    llm = _get_llm()
    system_prompt = PROMPTS["transcript_guidance_extraction"]["system"]
    user_prompt = PROMPTS["transcript_guidance_extraction"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"{user_prompt}\n\nTranscript:\n{state['transcript']}"
        ),
    ]
    response = llm.invoke(messages)
    return {"transcript_guidance_extracted": response.content}


def guid_validation_node(state: GraphState) -> dict:
    """Node 2.2 – validate extracted guidance against the full transcript."""
    llm = _get_llm()
    system_prompt = PROMPTS["guid_validation"]["system"]
    user_prompt = PROMPTS["guid_validation"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Transcript:\n{state['transcript']}\n\n"
                f"Extracted Guidance:\n{state['transcript_guidance_extracted']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"guid_validation_out": response.content}


# ---------------------------------------------------------------------------
# Branch 3: Key Message
# ---------------------------------------------------------------------------

def segment_extraction_node(state: GraphState) -> dict:
    """Node 3.1 – extract structured segment data from the transcript. (Structured Output)"""
    llm = _get_llm()
    system_prompt = PROMPTS["segment_extraction"]["system"]
    user_prompt = PROMPTS["segment_extraction"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Transcript:\n{state['transcript']}\n\n"
                f"Segment Data:\n{state['segment_data']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"segment_extraction_out": {"content": response.content}}


def context_retrieval_node(state: GraphState) -> dict:
    """Node 3.2 – retrieve contextual information using segment extraction output. (Structured Output)"""
    llm = _get_llm()
    system_prompt = PROMPTS["context_retrieval"]["system"]
    user_prompt = PROMPTS["context_retrieval"]["user"]
    segment_extraction_str = json.dumps(state.get("segment_extraction_out", {}))
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Transcript:\n{state['transcript']}\n\n"
                f"Segment Data:\n{state['segment_data']}\n\n"
                f"Segment Extraction Output:\n{segment_extraction_str}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"context_retrieval_out": {"content": response.content}}


def integrator_node(state: GraphState) -> dict:
    """Node 3.3a – integrate context retrieval and segment extraction results."""
    llm = _get_llm()
    system_prompt = PROMPTS["integrator"]["system"]
    user_prompt = PROMPTS["integrator"]["user"]
    context_retrieval_str = json.dumps(state.get("context_retrieval_out", {}))
    segment_extraction_str = json.dumps(state.get("segment_extraction_out", {}))
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Context Retrieval Output:\n{context_retrieval_str}\n\n"
                f"Segment Extraction Output:\n{segment_extraction_str}\n\n"
                f"Segment Items:\n{state['segment_items']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"integrator_out": response.content}


def key_messages_node(state: GraphState) -> dict:
    """Node 3.4a – produce key messages from the integrator output."""
    llm = _get_llm()
    system_prompt = PROMPTS["key_messages"]["system"]
    user_prompt = PROMPTS["key_messages"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Integrator Output:\n{state['integrator_out']}\n\n"
                f"Segment Data:\n{state['segment_data']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"key_messages_out": response.content}


def summarizer_node(state: GraphState) -> dict:
    """Node 3.3b – summarize context retrieval and segment extraction results."""
    llm = _get_llm()
    system_prompt = PROMPTS["summarizer"]["system"]
    user_prompt = PROMPTS["summarizer"]["user"]
    context_retrieval_str = json.dumps(state.get("context_retrieval_out", {}))
    segment_extraction_str = json.dumps(state.get("segment_extraction_out", {}))
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Context Retrieval Output:\n{context_retrieval_str}\n\n"
                f"Segment Extraction Output:\n{segment_extraction_str}\n\n"
                f"Segment Items:\n{state['segment_items']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"summarizer_out": response.content}


def briefing_key_messages_node(state: GraphState) -> dict:
    """Node 3.4b – produce briefing key messages from the summarizer output."""
    llm = _get_llm()
    system_prompt = PROMPTS["briefing_key_messages"]["system"]
    user_prompt = PROMPTS["briefing_key_messages"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Summarizer Output:\n{state['summarizer_out']}\n\n"
                f"Segment Data:\n{state['segment_data']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"briefing_key_messages_out": response.content}


# ---------------------------------------------------------------------------
# Branch 4: QA
# ---------------------------------------------------------------------------

def transcript_QA_extraction_node(state: GraphState) -> dict:
    """Node 4.1 – extract Q&A content from the transcript."""
    llm = _get_llm()
    system_prompt = PROMPTS["transcript_QA_extraction"]["system"]
    user_prompt = PROMPTS["transcript_QA_extraction"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"{user_prompt}\n\nTranscript:\n{state['transcript']}"
        ),
    ]
    response = llm.invoke(messages)
    return {"transcript_QA_extracted": response.content}


def second_QA_node(state: GraphState) -> dict:
    """Node 4.2 – perform a second-pass QA over the extracted QA content."""
    llm = _get_llm()
    system_prompt = PROMPTS["second_QA"]["system"]
    user_prompt = PROMPTS["second_QA"]["user"]
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"{user_prompt}\n\n"
                f"Extracted QA:\n{state['transcript_QA_extracted']}"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"second_QA_out": response.content}


# ---------------------------------------------------------------------------
# Aggregation & wrapping nodes
# ---------------------------------------------------------------------------

def output_template_node(state: GraphState) -> dict:
    """
    Merge node – waits for all four branches to complete, then generates the
    final AI summary using outputs from every branch.
    """
    llm = _get_llm()
    system_prompt = PROMPTS["output_template"]["system"]
    user_prompt = PROMPTS["output_template"]["user"]
    context = (
        f"FA Highlights:\n{state.get('fa_highlights_out', '')}\n\n"
        f"Guidance Validation:\n{state.get('guid_validation_out', '')}\n\n"
        f"Briefing Key Messages:\n{state.get('briefing_key_messages_out', '')}\n\n"
        f"Key Messages:\n{state.get('key_messages_out', '')}\n\n"
        f"QA:\n{state.get('second_QA_out', '')}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{user_prompt}\n\n{context}"),
    ]
    response = llm.invoke(messages)
    return {"ai_summary": response.content}


def wrapper_node(state: GraphState) -> dict:
    """
    Final node (no LLM). Wraps the ai_summary string into the required
    deeply nested JSON response structure.
    """
    final_response = {
        "outputs": [
            {
                "outputs": [
                    {
                        "results": {
                            "message": {
                                "text": state["ai_summary"]
                            }
                        }
                    }
                ]
            }
        ]
    }
    return {"final_response": final_response}
