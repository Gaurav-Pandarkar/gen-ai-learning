import streamlit as st
import requests

st.set_page_config(page_title="ChatGPT-style Chatbot", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #6A0DAD; 
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        .stButton button {
            background-color: white !important;
            color: #007BFF !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: bold !important;
        }
        .stButton button:hover {
            background-color: #e6e6e6 !important;
        }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

def new_chat():
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### 💬 Chat Options")
    st.button("New Chat", on_click=new_chat)

chat_container = st.container()

with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here...")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = requests.post(
            "http://127.0.0.1:5400/api/chat", 
            json={"message": user_input}
        )
        ai_message = response.json().get("response", "Error: No response")
    except Exception as e:
        ai_message = f"Error: {e}"

    st.session_state.messages.append({"role": "bot", "content": ai_message})

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑‍💬 User:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Bot:** {msg['content']}")
