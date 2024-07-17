import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

components.html(
    """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        const githubButton = document.querySelector('a[href="https://github.com/streamlit"]');
        if (githubButton) {
            githubButton.style.display = 'none';
        }
    });
    </script>
    """,
    height=0,
)

load_dotenv()
api_key = os.getenv("api_key")

# Define the system template for the prompt
system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

For every answer you give, show the website link you got the information from.

If you are directing a user to go to a certain page on the website, provide the link.
"""

# Create the prompt templates
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

# Define function to load and prepare documents
def load_and_prepare_documents():
    # Load data from the specified URL
    loader = WebBaseLoader(['https://www.ourdailybreadmot.com/', 'https://www.ourdailybreadmot.com/blog', 'https://www.ourdailybreadmot.com/about_us', 'https://www.ourdailybreadmot.com/board-of-directors', 'https://www.ourdailybreadmot.com/contact', 'https://www.ourdailybreadmot.com/training', 'https://www.ourdailybreadmot.com/get_involved', 'https://www.ourdailybreadmot.com/forms'])
    data = loader.load()

    # Split the loaded data into chunks
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=40)
    docs = text_splitter.split_documents(data)
    return docs

# Streamlit app
def main():
    st.title('Our Daily Bread MOT Q&A')

    # User inputs
    query = st.chat_input("Ask a question about ODB")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Set up paths
        ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        DB_DIR = os.path.join(ABS_PATH, "db")

        # Load and prepare documents
        docs = load_and_prepare_documents()

        # Create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Create a Chroma vector database from the documents
        vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings, persist_directory=DB_DIR)
        vectordb.persist()

        # Create a retriever from the Chroma vector database
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Use a ChatOpenAI model
        llm = ChatOpenAI(model_name='gpt-4-turbo', temperature=0.85, api_key=api_key)

        # Create a RetrievalQA chain from the model and retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # Run the prompt and return the response
        response = qa(query)
        response_content = response.get('result')

        st.session_state.messages.append({"role": "assistant", "content": response_content})

        with st.chat_message("assistant"):
            st.markdown(response_content)

if __name__ == '__main__':
    main()
