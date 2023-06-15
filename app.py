import tempfile
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import time

OpenAI_API_KEY = st.secrets["openai_api_key"]

embeddings = OpenAIEmbeddings(openai_api_key=OpenAI_API_KEY)

persist_directory = 'pdf_persist'
collection_name = 'pdf_collection'

llm = OpenAI(temperature=0, openai_api_key=OpenAI_API_KEY, model_name="gpt-3.5-turbo-0613", max_tokens=2048)

chain = load_qa_chain(llm, chain_type='refine')

vectorstore = None

def load_pdf(pdf_path):
    return PyMuPDFLoader(pdf_path).load()

st.title("PDFChatBot")

with st.container():
    upload_file = st.file_uploader("Please choose your PDF file", type='pdf')
    if upload_file is not None:
        if upload_file.size > 6 * 1024 * 1024:  # 6MB
            st.error("File size exceeds the limit of 6MB.")
        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                with open(temp_path, 'wb') as f:
                    f.write(upload_file.getbuffer())
                docs = load_pdf(temp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        vectorstore.persist()

        st.write("Finished")

with st.container():
    question = st.text_input("Question")
    if vectorstore is not None and question is not None and question != "":
        with st.spinner("AI is thinking..."):
            time.sleep(2)  # Simulate waiting for the answer
            docs = vectorstore.similarity_search(question, 3, include_metadata=True)
            answer = chain.run(input_documents=docs, question=question)
            st.write(answer)
