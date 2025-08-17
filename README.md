---
title: Clausenaut
emoji: 🔥
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
short_description: 'insurance Document analyzer '
license: mit
---
# 🚢 ClauseNaut  
**An LLM-Powered Document Query System for Intelligent Claim Adjudication**

ClauseNaut is a **Retrieval-Augmented Generation (RAG)** system designed to analyze policy documents and make informed decisions on user claims.  
It features a **user-friendly web interface** built with [Gradio](https://www.gradio.app/), allowing users to upload documents (PDFs, DOCX), ask complex questions in natural language, and receive a **structured JSON-based decision** with justifications.
<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/18318d09-1365-435b-ab0d-139397c4aab6" />

---

## ✨ Key Features
- 📄 **Multi-Format Document Support** – Upload and process `.pdf` and `.docx` files.  
- 🧠 **Intelligent Query Structuring** – Uses **Google’s Gemini model** to parse natural language queries into structured key-value pairs.  
- 🔍 **Semantic Search** – Employs **LangChain + Chroma vector store** for retrieving the most relevant clauses.  
- 🤖 **AI-Powered Adjudication** – Gemini LLM compares clauses against the user’s claim to render a decision.  
- 📝 **Structured Output** – Returns a JSON object containing:
  - Decision (Approved/Rejected)  
  - Approved amount (if applicable)  
  - Detailed justification citing source clauses  
- 💻 **Simple Web UI** – A seamless **3-step Gradio interface** for claim adjudication.  

---

## ⚙️ How It Works (The RAG Pipeline)

### Phase 1: Knowledge Base Creation
1. **Document Upload** – Upload one or more policy documents.  
2. **Document Loading & Chunking** – LangChain loaders split documents into manageable chunks.  
3. **Embedding & Indexing** – Chunks converted into embeddings using HuggingFace → stored in **Chroma DB**.  

✅ Knowledge base is ready for queries.  

### Phase 2: Claim Processing
1. **User Query** – e.g., *“46-year-old male, knee surgery in Pune, 3-month-old insurance policy”*.  
2. **Query Structuring** – First LLM call extracts fields into JSON (Age, Procedure, Location, etc.).  
3. **Context Retrieval** – Semantic search fetches relevant policy clauses.  
4. **Final Decision Generation** – Second LLM call adjudicates claim → produces structured JSON with verdict & justification.  

---

## 🚀 Getting Started

### Prerequisites
- Python **3.8+**
- Google Gemini API Key ([Get it here](https://aistudio.google.com/))

---

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ClauseNaut.git
cd ClauseNaut
```

2. Set Up Virtual Environment
Windows
```bash
python -m venv venv
venv\Scripts\activate
```
macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
4. Install Dependencies
```bash
gradio
google-generativeai
langchain-community
pypdf
docx2txt
sentence-transformers
chromadb
warnings
```
Then install:
```bash
pip install -r requirements.txt
```
6. Configure Environment Variables
```bash
Create a .env file in the project root:
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
Or set it as a system environment variable.
```
8. Run the Application
```bash
python app.py
```
Open browser → http://127.0.0.1:7860

📖 Usage Guide
Upload Documents – Add PDF/DOCX policy files.
Wait for Knowledge Base – Confirm readiness status.
Define Extraction Fields – Default fields are provided (can be customized).
Ask a Claim Query – Enter claim details in natural language.
Submit – Click Submit Query.
Review Output – View final structured JSON verdict in the Final Decision panel.


🛠️ Tech Stack
Frontend/UI: Gradio
LLM: Google Gemini
Frameworks: LangChain, HuggingFace Embeddings
Vector DB: Chroma
Document Parsing: PyPDF, docx2txt


📌 Roadmap (Future Enhancements)
✅ Add support for Excel & TXT files
✅ Multi-user session handling
🚧 Integration with hospital/insurance APIs
🚧 Advanced claim analytics dashboard


🤝 Contributing
Contributions are welcome!
Fork this repo
Create a feature branch (git checkout -b feature-name)
Commit your changes
Submit a PR 🎉


📜 License
This project is licensed under the MIT License.
Made with ❤️ by Eswar

👉 This version is clean, professional, and visually appealing for GitHub.  

Do you also want me to **add a small architecture diagram (RAG pipeline workflow)** inside the README using Markdown + Mermaid, so it looks more visually descriptive?
