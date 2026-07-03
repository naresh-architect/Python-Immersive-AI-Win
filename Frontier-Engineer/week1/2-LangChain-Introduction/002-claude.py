from dotenv import load_dotenv
import os
import warnings

from langchain_anthropic import ChatAnthropic

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv("/Users/nareshchaurasia/nc/PYTHON-ARCHITECT/Python-Immersive-AI/.env_mac")

prompt = "What is API key?"


def create_claude_llm(model_name="claude-sonnet-4-6", **kwargs):

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found")

    llm = ChatAnthropic(
        model=model_name,
        anthropic_api_key=api_key,
        **kwargs
    )

    return llm


# Test it
llm_claude = create_claude_llm()

response = llm_claude.invoke(prompt)

print(response.content)