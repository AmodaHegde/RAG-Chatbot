import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import langchain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import jwt

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
    store_answers(response['chat_history'])

def store_answers(chat_history):
    # Create a file name based on date and time
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_name = f"convo_{current_date}.txt"

    # Write chat history to the file
    i=0
    with open(file_name, 'a') as file:
        for message in chat_history:
            if hasattr(message, 'content'):
                role = "User" if (i%2==0) else "Bot"
                file.write(f"{role}: {message.content}\n")
                i+=1

def redirectPage():
    # Define the expected secret key
    SECRET_KEY = 'ABCD'
    # Retrieve the token from the request
    received_token = st.experimental_request_metadata().get("query_string_params", {}).get("token", "")
    try:
        # Decode and verify the token
        payload = jwt.decode(received_token, SECRET_KEY, algorithms=["HS256"])
    
        # Validate the payload (e.g., check user ID)
        user_id = payload.get("userId")
    
        if user_id != '123456':
            st.write("Invalid user!")
            st.stop()

        # Continue with the Streamlit application
        presentScreen()
    
    except jwt.ExpiredSignatureError:
        st.write("Token has expired!")
        st.stop()

    except jwt.InvalidTokenError:
        st.write("Invalid token!")
        st.stop()
        
def presentScreen():
    load_dotenv()
    st.set_page_config(page_title="AI Legal Contract Analysis")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("AI ANALYSE")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader(
            "Upload your Legal Contract PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

def main():
    redirectPage()

if __name__ == '__main__':
    main()
