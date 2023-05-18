import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai.error import AuthenticationError

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def parseCSV(uploaded_file, api_key):
    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    agent = create_csv_agent(OpenAI(openai_api_key=api_key,temperature=0), uploaded_file, verbose=False)
    user_question = st.text_input("Ask Rejibo a question about your CSV:")
    if user_question is not None and user_question != "":
        with st.spinner(text="In progress ..."):
            st.write(agent.run(user_question))

def parsePDF(uploaded_file, api_key):
    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    pdf_reader = PdfReader(uploaded_file) #just pdf reader
    text = ""
    for page in pdf_reader.pages: #loop through pages and extract text
        text += page.extract_text()

    # split into chunks bennycheung.github.io/ask-a-book-question-with-langchain-openai
    #splitujeme do chunkov aby sa lepsie pracovalo modelu stym
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, #1000 characters
        chunk_overlap=200, #ak 1000 konci niekde v strede tak dalsi sa bude zacinat 200 characters predtym. Aby sa nesplitli myslienky v strede
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key) #vector representation of text.
    knowledge_base = FAISS.from_texts(chunks, embeddings) #FAISS sluzi na efektivne hladanie podobnosti
    #FAISS = Facebook AI Similiarity Search cize sluzi na similiarity search in knowledge base
    # show user input
    user_question = st.text_input("Ask Rejibo a question about your PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question) #semantic similiarity search
        #hlada vektory ktore su podobne a v daka tomu vieme najst take chunky
        #  ktore obsahuju info ktore potrebujeme
        #docs obsahuje uz len tie chunky kde su data ktore chceme a tie vyuzijeme na opytanie sa jazykoveho modelu
        llm = OpenAI(openai_api_key=api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
    
        st.write(response)