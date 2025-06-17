**PDF Question Answering Chatbot**  
A RAG application that allows you to ask questions from a PDF file using Generative AI. The app leverages LangChain, OpenAI's LLMs, and FAISS vector store for semantic search and contextual responses.  

🧠 **Features**  
💬 Chat interface for real-time question answering  
🔍 Retrieves answers directly from PDF content  
⚡ Fast and efficient using FAISS vector store  
🧾 Context-aware LLM prompt with gpt-4o-mini  
💡 Beautiful and user-friendly UI with custom chat styles  

🗂️ **Folder Structure**  
├── app.py                # Main Streamlit app  
├── .env                  # API keys file  
├── TheHundred-pageMachineLearning.pdf  # PDF source  
└── README.md             # Project documentation  

🚀 **Getting Started**  
1. Clone the Repository  
git clone https://github.com/yourusername/pdf-chatbot.git  
cd Expleogroup-Assignment  

2. Create and Activate Virtual Environment  
python -m venv venv  
source venv/bin/activate  

3. Install Required Dependencies  
pip install -r requirements.txt  

4. Add the keys in .env File  
OPENAI_API_KEY=your_openai_api_key  
LANGCHAIN_API_KEY=your_langchain_api_key  

5. Run the App  
streamlit run app.py

**Note: Allow sometime to load the pdf as it is contains 150 pages**

