import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import config
from prompt import build_messages

load_dotenv()

_llm = ChatGroq(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    max_tokens=config.LLM_MAX_TOKENS,
    api_key=os.environ.get("GROQ_API_KEY"),
)


def get_llm():
    return _llm


def generate_answer(query, chunks):
    llm = get_llm()
    messages = build_messages(query, chunks)
    response = llm.invoke(messages)
    return response.content
