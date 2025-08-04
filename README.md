ClauseNaut: LLM-Powered Document Query System

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system designed to answer natural language queries based on a collection of local documents (PDFs and DOCX). It's specifically tailored for a use case like insurance claim adjudication, where structured data needs to be extracted from a query and evaluated against policy documents.

The system uses Google's gemini-1.5-flash-latest model for its advanced reasoning capabilities and features a dynamic, two-step query process.

✨ Key Features

Multi-Format Document Support: Ingests and processes both .pdf and .docx files.
Persistent Knowledge Base: Uses ChromaDB to create and store a searchable vector database of your documents, so the indexing process only needs to be run once.
Advanced RAG Pipeline:
Dynamic Query Structuring: An initial LLM call intelligently extracts key information from a user's free-form query into a structured JSON object. The fields to extract are fully customizable.
Contextual Retrieval: Uses sentence-transformers embeddings to find the most relevant document passages related to the query.
Justified Decision Making: A final LLM call analyzes the retrieved context and the structured query to generate a reasoned "Approved" or "Rejected" decision, complete with citations from the source documents.
Business Logic Override: Includes a specific example of a critical override rule (e.g., automatically approving "knee surgery") to demonstrate how to enforce custom business logic.
Colab-Ready: Designed to be run seamlessly in a Google Colab environment with secure API key management using Colab Secrets.


⚙️ How It Works

The system operates in two main phases:

Phase 1: Indexing индексирование
This is a one-time setup process that builds the knowledge base.
Load Documents: All .pdf and .docx files from the /documents directory are loaded.
Chunk Documents: The documents are split into smaller, overlapping text chunks using RecursiveCharacterTextSplitter. This allows the model to find specific, relevant context.
Generate Embeddings: Each chunk is converted into a numerical vector (an "embedding") using the all-MiniLM-L6-v2 model. These embeddings capture the semantic meaning of the text.
Store in VectorDB: The chunks and their corresponding embeddings are stored in a ChromaDB vector store, which is persisted to the disk in the /chroma_db directory.

Phase 2: Querying ❓
This phase is executed for every user query.
Input Query: The user provides a natural language query (e.g., "A 45-year-old man needs knee surgery in Mumbai.").
Structure Query: The first LLM call extracts predefined entities (e.g., age, medical procedure, city) from the query and structures them into a JSON object.
Retrieve Context: The system uses the user's query and the structured JSON to search the ChromaDB vector store for the most relevant document chunks (the "context").
Generate Final Decision: The second LLM call receives the retrieved context and the structured query. It analyzes everything, applies any special rules (like the "knee surgery" override), and generates the final JSON response with a decision and justification.
🚀 Getting Started
This script is optimized for Google Colab.

Prerequisites
A Google Account.
A Google AI Studio API key. You can get one from Google AI Studio.
Setup and Execution
Create a Colab Notebook: Open Google Colab and create a new notebook.
Add Your API Key:
In the left sidebar, click the key icon (Secrets).
Add a new secret:
Name: GOOGLE_API_KEY
Value: Paste your actual Google AI Studio API key here.
Make sure the "Notebook access" toggle is enabled.
Create Documents Folder:
In the left sidebar, click the folder icon (Files).
Click the "New folder" icon and name the folder documents.
Upload Your Files:
Click the three dots next to the documents folder and select "Upload".
Upload the PDF and/or DOCX files you want the system to learn from.
Run the Script:
Copy the entire Python script provided in the prompt into a single cell in your Colab notebook.
Run the cell by pressing Shift + Enter or clicking the play button.
Usage
Once the script finishes the "Indexing" phase, it will start an interactive session:
Define Extraction Fields: The first prompt will ask you to define the information you want to extract from user queries.
👉 Enter the comma-separated fields you want to extract (e.g., age, gender, medical procedure, city):
Example Input: medical procedure, policy holder age, city of treatment
Enter Your Query: After setting the fields, you can start asking questions.
Enter your claim query (or type 'quit' to exit):
Example Input: My claim is for a knee surgery. The policy holder is 52 years old and the treatment was in Delhi.
Receive the Output: The system will process the query through the full pipeline and print a final JSON object containing the decision and justification.
Type quit and press Enter to end the session.
🛠️ Customization
LLM Model: You can change the model by modifying the model name in the get_structured_query and get_final_decision functions (e.g., from gemini-1.5-flash-latest to gemini-1.5-pro-latest).
Prompts: The core logic is defined in the prompts within the get_structured_query and get_final_decision functions. You can edit these prompts to change the AI's behavior, persona, or output format.
Embedding Model: To use a different embedding model, change the model_name in the HuggingFaceEmbeddings initializer.
Retriever Settings: You can adjust the number of retrieved documents by changing the k value in vector_store.as_retriever(search_kwargs={"k": 5}).
📦 Dependencies
The script will automatically install the following Python libraries:
langchain
google-generativeai
langchain_community
sentence-transformers
chromadb
pypdf
python-docx
