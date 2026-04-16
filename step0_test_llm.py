import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
import ulid

load_dotenv()

def generate_session_id():
    team = os.getenv("orchestron", "team").replace(" ", "-")
    return f"{team}-{ulid.new().str}"

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=200,
)

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

@observe()
def test_llm_call(session_id):
    handler = CallbackHandler()
    response = model.invoke(
        [HumanMessage(content="Say hello in one sentence.")],
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id}
        }
    )
    return response.content

if __name__ == "__main__":
    session_id = generate_session_id()
    print("Session ID:", session_id)
    print("LLM response:", test_llm_call(session_id))
    langfuse_client.flush()