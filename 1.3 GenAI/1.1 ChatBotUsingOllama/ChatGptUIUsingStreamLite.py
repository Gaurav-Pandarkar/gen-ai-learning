import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOllama(
    model="llama3.2",      
    max_tokens=500,
    temperature=0.7,
)

st.set_page_config(
    page_title="💬 ChatBot using Ollama",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 ChatBot powered by Ollama")
st.caption("Built with LangChain + Streamlit | Local LLM running via Ollama")


if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant, answer clearly and politely.")
    ]

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

user_input = st.chat_input("Type your message and press Enter...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.messages)
            st.markdown(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))
