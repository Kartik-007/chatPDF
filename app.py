import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectordatabase_openAi(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectordatabase = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectordatabase

def get_vectordatabase_hf(text_chunks):
    model_name = "hkunlp/instructor-xl"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )
    vectordatabase = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectordatabase

def get_conversation_thread(vectordatabase):
    llm = OpenAi()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_thread = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordatabase.as_retriever(),
        memory=memory
    )
    return conversation_thread


def main():
    load_dotenv()
    st.set_page_config(page_title="chatPDF", page_icon=":books")
    st.header("chatPDF :books:")
    st.text_input("Ask a question...")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files= True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get text from pdfs
                raw_text = get_pdf_text(pdf_docs)

                #split huge text into chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector database
                vectordatabase = get_vectordatabase_hf(text_chunks)

                #create a conersation thread
                conversation_thread = get_conversation_thread(vectordatabase)



if __name__ == '__main__':
    main()