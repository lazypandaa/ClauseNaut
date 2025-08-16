ClauseNaut 🚢
An LLM-Powered Document Query System for Intelligent Claim Adjudication

ClauseNaut is a Retrieval-Augmented Generation (RAG) system designed to analyze policy documents and make informed decisions on user claims. It features a user-friendly web interface built with Gradio, allowing users to upload documents (PDFs, DOCX), ask complex questions in natural language, and receive a structured, justified decision in JSON format.

✨ Key Features
📄 Multi-Format Document Support: Upload and process .pdf and .docx files to build a custom knowledge base.
🧠 Intelligent Query Structuring: Uses Google's Gemini model to parse natural language queries and extract key-value information.
🔍 Semantic Search: Leverages LangChain and a Chroma vector store to find the most relevant clauses from your documents.
🤖 AI-Powered Adjudication: The Gemini LLM analyzes the retrieved clauses against the user's claim to render a final, reasoned decision.
📝 Structured Output: Delivers the verdict as a clean JSON object, including the decision, approved amount, and a detailed justification citing specific source clauses.
💻 Easy-to-Use Web UI: A simple, three-step interface built with Gradio for a seamless user experience.
⚙️ How It Works (The RAG Pipeline)
ClauseNaut operates in two main phases:

Phase 1: Knowledge Base Creation
Document Upload: You upload one or more policy documents through the Gradio interface.
Document Loading & Chunking: LangChain loaders process the files, and the text is split into smaller, manageable chunks.
Embedding & Indexing: Each chunk is converted into a numerical vector using HuggingFace embeddings and stored in a Chroma vector database. This knowledge base is now ready for queries.
Phase 2: Claim Processing
User Query: A user submits a claim query in natural language (e.g., "46-year-old male, knee surgery in Pune, 3-month-old insurance policy").
Query Structuring: The first LLM call extracts predefined fields (e.g., Age, Procedure, Location) from the query, converting it into a structured JSON object.
Context Retrieval: The system performs a semantic search on the vector store using the query to find the most relevant document chunks (the "context").
Final Decision Generation: A second, more complex prompt is sent to the LLM. This prompt includes the retrieved context and the structured query, asking the AI to act as an adjudicator and generate a final JSON output with a decision and justification.
🚀 Getting Started
Follow these steps to run ClauseNaut on your local machine.

Prerequisites
Python 3.8+
A Google Gemini API Key. You can get one from Google AI Studio.
1. Clone the Repository
git clone https://github.com/your-username/ClauseNaut.git
cd ClauseNaut

2. Set Up a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Create a requirements.txt file with the following content:

gradio
google-generativeai
langchain-community
pypdf
docx2txt
sentence-transformers
chromadb
warnings

Then, install the packages:

pip install -r requirements.txt

4. Configure Environment Variables
Create a file named .env in the root of your project directory and add your Google Gemini API key. The script is set up to load this key automatically.

GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

Alternatively, you can set it as a system environment variable.

5. Run the Application
Launch the Gradio web server with the following command (assuming your script is named app.py):

python app.py

Open your web browser and navigate to the local URL provided in the terminal (usually http://127.0.0.1:7860).

📖 Usage Guide
Upload Documents: Drag and drop your PDF or DOCX policy files into the file uploader. Wait for the "Knowledge Base Status" to confirm that it's ready.
Define Extraction Fields: The Comma-separated fields to extract textbox tells the LLM what information to look for in the user's query. The default values are a good starting point.
Ask a Question: Enter the user's claim details in the Claim Query box.
Submit: Click the Submit Query button.
Review Decision: The final, structured JSON decision will appear in the "Final Decision" box on the right.
🛠️ Customization
LLM Model: You can easily switch the Gemini model by changing the model name in the script (e.g., from 'gemini-1.5-flash-latest' to 'gemini-1.5-pro-latest').
Embedding Model: The embedding model can be changed in the build_knowledge_base function. The current default is sentence-transformers/all-MiniLM-L6-v2.
Prompts: The core logic is in the prompts! Feel free to modify the prompts in get_structured_query and get_final_decision to change the LLM's behavior or output format.
