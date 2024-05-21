import os

import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

st.title("Chat with Document")
upload_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'docx'])
add_file = st.button("Add File", on_click=clear_history)

if upload_file and add_file:
    with st.spinner("Reading, Chunking and Embedding..."):
        bytes_data = upload_file.read()
        file_name = os.path.join('./', upload_file.name)
        with open(file_name, 'wb') as f:
            f.write(bytes_data)

        name, extension = os.path.splitext(file_name)

        if extension == '.pdf':
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_name)
        elif extension == '.docx':
            from langchain.document_loaders import Docx2TextLoader
            loader = Docx2TextLoader(file_name)
        elif extension == '.txt':
            loader = TextLoader(file_name)
        else:
            st.error("File format not supported")
            # st.stop()
        
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()

        vector_store = Chroma.from_documents(chunks, embeddings)

        # llm = OpenAI(temperature=0)
        llm = ChatOpenAI(model="gpt-4o", temperature=1)

        retriever = vector_store.as_retriever()

        # chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

        crc = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
        st.session_state['crc'] = crc
        st.success("File added successfully")

question = st.text_input("Ask a question about the constitution")

if question:
    # response = chain.run(question)
    if 'crc' in st.session_state:
        crc = st.session_state['crc']
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        response = crc.run({'question': question, 'chat_history': st.session_state['history']})
        
        st.session_state['history'].append((question, response))
        st.write(response)

        # st.write(st.session_state['history'])
        for prompts in st.session_state['history']:
            st.write("question: ", prompts[0])
            st.write("answer: ", prompts[1])
