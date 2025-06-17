import os
import streamlit as st
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Assignment"

# Define resume file path
SOURCE_PATH = "./TheHundred-pageMachineLearning.pdf"  # Ensure this file exists in the folder

# App title
st.markdown("<h1 style='text-align: center;'>Expleogroup Assignment</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Ask any question from PDF and I will answer you.</p>", unsafe_allow_html=True)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "thinking" not in st.session_state:
    st.session_state.thinking = False

# Process the resume if not already processed
if os.path.exists(SOURCE_PATH) and st.session_state.retrieval_chain is None:
    with st.spinner("⏳ Processing data..."):
        # Load and process the document
        loader = PyPDFLoader(SOURCE_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        
        # Define LLM and prompt
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            You are an AI Assistant, who answers the questions based on the context.
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            
            Question: {input}
            """
        )
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Store in session state
        st.session_state.retrieval_chain = retrieval_chain
        
        st.success("✅ I am ready to answer your questions 🎉")
else:
    if not os.path.exists(SOURCE_PATH):
        st.error(f"File not found at {SOURCE_PATH}. Please ensure the file is available.")

# Add custom CSS for chat alignment with dark mode compatibility and icons
st.markdown("""
<style>
/* Container for all messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 10px;
}
/* Message row layout */
.message-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 8px;
}
/* User message styling */
.user-row {
    justify-content: flex-end;
}
.user-icon {
    margin-left: 12px;
    background-color: #1976D2;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
}
.user-message-content {
    background-color: #1E88E5;
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 12px 16px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
/* Assistant message styling */
.assistant-row {
    justify-content: flex-start;
}
.assistant-icon {
    margin-right: 12px;
    background-color: #424242;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
}
.assistant-message-content {
    background-color: #424242;
    color: white;
    border-radius: 18px 18px 18px 0;
    padding: 12px 16px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
/* Thinking indicator style */
.thinking-row {
    justify-content: flex-start;
}
.thinking-bubble {
    background-color: #424242;
    color: white;
    border-radius: 18px;
    padding: 12px 16px;
    display: inline-block;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.thinking-dots {
    display: flex;
    align-items: center;
    height: 16px;
}
.dot {
    height: 8px;
    width: 8px;
    margin: 0 2px;
    background-color: white;
    border-radius: 50%;
    opacity: 0.7;
    animation: pulse 1.5s infinite ease-in-out;
}
.dot:nth-child(1) {
    animation-delay: 0s;
}
.dot:nth-child(2) {
    animation-delay: 0.3s;
}
.dot:nth-child(3) {
    animation-delay: 0.6s;
}
@keyframes pulse {
    0%, 100% { transform: scale(0.8); opacity: 0.7; }
    50% { transform: scale(1.2); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Create container for chat display
chat_container = st.container()

# Function to display messages
def display_messages():
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display all stored messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-row user-row">
                    <div class="user-message-content">
                        {message["content"]}
                    </div>
                    <div class="user-icon">
                        👤
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-row assistant-row">
                    <div class="assistant-icon">
                        🤖
                    </div>
                    <div class="assistant-message-content">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show thinking indicator if the bot is thinking
        if st.session_state.thinking:
            st.markdown("""
            <div class="message-row thinking-row">
                <div class="assistant-icon">
                    🤖
                </div>
                <div class="thinking-bubble">
                    <div class="thinking-dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display current messages
display_messages()

# Input for user question
if prompt := st.chat_input("type your question here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Set thinking state to true and update display
    st.session_state.thinking = True
    st.rerun()

# Generate response if thinking is active
if st.session_state.thinking and st.session_state.retrieval_chain is not None:
    # Generate response
    result = st.session_state.retrieval_chain.invoke({"input": st.session_state.messages[-1]["content"]})
    answer = result['answer']
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Set thinking state to false
    st.session_state.thinking = False
    
    # Force a rerun to display the updated messages
    st.rerun()

st.markdown("---")
st.markdown("👨‍💻 Developed with ❤️ using OpenAI, LangChain & Streamlit")