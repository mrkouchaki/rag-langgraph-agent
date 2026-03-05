import streamlit as st
import requests

API_URL = "http://localhost:8080/ask"

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Research Assistant")
st.write("Ask me anything! I can search the web and my knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt, "max_steps": 6}
                response = requests.post(API_URL, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer", "No answer found.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to backend. Is FastAPI running on port 8080?")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
