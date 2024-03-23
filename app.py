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
from htmlTemplates import css, bot_template, user_template

# Extracts text from a list of PDF documents.
# Each PDF is read page by page, and text is extracted and concatenated.
# Input: pdf_docs - A list of PDF files uploaded by the user.
# Output: A single string containing all the text extracted from the PDFs.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Splits a large string of text into manageable chunks.
# This is useful for processing large texts in pieces, especially for embedding or analysis.
# Input: raw_text - The complete text to be split into chunks.
# Output: A list of text chunks, split according to specified parameters.
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

# Generates embeddings for text chunks using an OpenAI model and stores them in a FAISS database.
# This allows for efficient similarity search among text chunks.
# Input: text_chunks - A list of text chunks for which embeddings are to be generated.
# Output: A FAISS database object containing the embeddings of the text chunks.
def get_vectordatabase_openAi(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectordatabase = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectordatabase

# Generates embeddings for text chunks using a Hugging Face model and stores them in a FAISS database.
# This method is similar to `get_vectordatabase_openAi` but utilizes a free Hugging Face model.
# Input: text_chunks - A list of text chunks for which embeddings are to be generated.
# Output: A FAISS database object containing the embeddings of the text chunks.
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

# Creates a conversational retrieval chain.
# This function initializes a conversational model with memory and a retriever to handle user queries.
# The retriever uses a vector database to find relevant text chunks based on the query.
# Input: vectordatabase - A FAISS database object containing text chunk embeddings.
# Output: A ConversationalRetrievalChain object that can handle conversational queries with contextual memory.
def get_conversation_thread(vectordatabase):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_thread = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordatabase.as_retriever(),
        memory=memory
    )
    return conversation_thread


#Handles the users input 
def handle_userinput(user_input):
    # Process the user input by updating the conversation thread in the session state
    response = st.session_state.conversation_thread({'question': user_input})
    # Update the chat history in the session state with the response
    st.session_state.chat_history = response['chat_history']

    # Iterate over the chat history to display each message
    for i, message in enumerate(st.session_state.chat_history):
        # For even indices (user messages), use the user template
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        # For odd indices (bot responses), use the bot template
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# Define the main function of the application
def main():
    # Load environment variables from a .env file
    load_dotenv()
    
    # Set the configuration for the Streamlit page
    st.set_page_config(page_title="chatPDF", page_icon=":books")
    # Include custom CSS styles
    st.write(css, unsafe_allow_html=True)

    # Initialize conversation thread in session state if not already present
    if "conversation_thread" not in st.session_state:
        st.session_state.conversation_thread = None

    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Display the application header
    st.header("chatPDF :books:")
    # Create an input box for user questions
    user_input = st.text_input("Ask a question...")

    # Handle user input if it is provided
    if user_input:
        handle_userinput(user_input)

    # Use Streamlit's sidebar to provide additional options
    with st.sidebar:
        st.subheader("Your documents")
        # Allow users to upload PDF documents
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        # Process the uploaded PDFs when the 'Process' button is clicked
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract text from the uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split the extracted text into manageable chunks
                text_chunks = get_text_chunks(raw_text)

                # Create a vector database from the text chunks for searching
                vectordatabase = get_vectordatabase_openAi(text_chunks)

                # Create a conversation thread based on the vector database
                st.session_state.conversation_thread = get_conversation_thread(vectordatabase)

# Execute the main function when the script is run directly
if __name__ == '__main__':
    main()