import os

import streamlit as st
from dotenv import load_dotenv

# Load env vars from .env (GROQ_API_KEY, etc.)
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Playground (LangChain)",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Agentic RAG Playground (LangChain)")

if os.getenv("GROQ_API_KEY"):
    st.caption("🔑 GROQ_API_KEY loaded from .env")
else:
    st.warning(
        "GROQ_API_KEY not found in environment. "
        "Set it in your .env file if you want to use GROQ."
    )

st.markdown(
    """
Welcome!  

Use the sidebar to navigate:

 **Chatbot Q&A** – talk with your agentic RAG chatbot.
 **RAG Evaluation** – evaluate the quality of your RAG system with RAGAS metrics.
    """
)


st.info("➡️ Select a page from the sidebar to get started.")
