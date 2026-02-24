"""
Prompt configuration for all LLM nodes.

Each entry contains a "system" prompt and a "user" prompt template.
Leave the strings empty here; fill them in before running the application.
The user prompt templates may reference state variables that the corresponding
node will append as additional context at call time.
"""

PROMPTS: dict[str, dict[str, str]] = {
    # Branch 1: Financial Highlights
    "transcript_fa_extraction": {"system": "", "user": ""},
    "fa_highlights": {"system": "", "user": ""},

    # Branch 2: Guidance
    "transcript_guidance_extraction": {"system": "", "user": ""},
    "guid_validation": {"system": "", "user": ""},

    # Branch 3: Key Message
    "segment_extraction": {"system": "", "user": ""},
    "context_retrieval": {"system": "", "user": ""},
    "integrator": {"system": "", "user": ""},
    "key_messages": {"system": "", "user": ""},
    "summarizer": {"system": "", "user": ""},
    "briefing_key_messages": {"system": "", "user": ""},

    # Branch 4: QA
    "transcript_QA_extraction": {"system": "", "user": ""},
    "second_QA": {"system": "", "user": ""},

    # Aggregation
    "output_template": {"system": "", "user": ""},
}
