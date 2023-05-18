import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def parseCSV(uploaded_file, api_key):
    agent = create_csv_agent(OpenAI(openai_api_key=api_key,temperature=0), uploaded_file, verbose=False)
    user_question = st.text_input("Ask a question about your CSV:")
    if user_question is not None and user_question != "":
        with st.spinner(text="In progress ..."):
            st.write(agent.run(user_question))

def parsePDF(uploaded_file, api_key):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
    
        st.write(response)