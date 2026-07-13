from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chat(system_prompt: str, user_prompt: str) -> str:
    """
    Single non-streaming call. generate_eval_dataset.py imports this to
    produce the "ground_truth" reference answer for each logged question,
    using the same context that was retrieved for the real answer.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = _model.invoke(messages)
    return response.content