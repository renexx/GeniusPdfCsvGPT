from dotenv import load_dotenv
import streamlit as st
import os
from components.sidebar import sidebar
from functions import parseCSV, parsePDF

sidebar()

def clear_submit():
    st.session_state["submit"] = False

def main():
    load_dotenv()
    api_key = st.session_state.get("OPENAI_API_KEY")
    if(api_key):
        st.header("Ask Rejibo 💬 about your PDF, CSV")
        api_key = st.session_state.get("OPENAI_API_KEY")
        
        # upload file
        uploaded_file = st.file_uploader("Upload a pdf, docx, or txt file", type=["pdf","csv"], help="Scanned documents are not supported yet!",on_change=clear_submit)

        
        # extract the text
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                parsePDF(uploaded_file,api_key)
            elif uploaded_file.name.endswith(".csv"):
                parseCSV(uploaded_file,api_key)
    else:
        st.header("Hello, please enter your OpenAI API key 🔑 into the input field in the sidebar")
        st.markdown(
            "1. Enter your OpenAI API key 🔑\n"  # noqa: E501
            "2. Upload a pdf, csv 📄\n"
            "3. Ask a question based on the document 💬\n"
        )

if __name__ == '__main__':
    main()
