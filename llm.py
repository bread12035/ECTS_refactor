import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def _get_llm() -> ChatOpenAI:
    """
    Instantiate and return the shared LLM client.

    Configuration is read from the .env file:
        OPENAI_API_KEY  – API key for the LLM provider.
        OPENAI_API_BASE – Base URL for the LLM provider endpoint.
    """
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
