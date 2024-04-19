import streamlit as st

st.set_page_config(
    page_title="Chatbot by CT",
    page_icon="🤖"
)

st.title("🏠 I can help you with your questions! We start ...")
st.write("📄 Page RAG pdfs will help you answer the content in 1 or more pdf files.")
st.write("🌐 Page RAG website will help you answer content on any website.")
st.sidebar.success("Select a page above.")