📂 Domain-Specific RAG using Google Drive + MistralAI

A Streamlit Application with FAISS Vector Search & Google Drive Document Ingestion

This project is a Retrieval-Augmented Generation (RAG) application built with Streamlit, designed to ingest documents from Google Drive, chunk them, embed them using SentenceTransformers, index them with FAISS, retrieve relevant context, and finally generate an answer using MistralAI.

🚀 Features
✅ Google Drive Integration

Authenticate using OAuth 2.0 (PyDrive)

Fetch documents from any Google Drive folder

Supports:

Google Docs

Text files

PDFs (basic extraction via PyDrive content export)

✅ RAG Pipeline

Smart text chunking with overlap

SBERT embeddings (all-MiniLM-L6-v2)

FAISS Vector Search

Query-based document ranking

✅ MistralAI Integration

Uses the mistralai client

Supports models like:

mistral-medium

(or any other model ID)

✅ Streamlit Web App

Clean UI

Sidebar configuration

Real-time querying

Retrieved context shown for transparency

Handles missing dependencies gracefully

📁 Project Structure
streamlit_rag_gdrive.py  
client_secrets.json  (required for Google OAuth)
README.md

🛠️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2️⃣ Install dependencies
pip install -r requirements.txt


Recommended libs:

streamlit
sentence-transformers
faiss-cpu
pydrive
mistralai
numpy

🔑 Google Drive Setup

Go to Google Cloud Console → APIs & Services

Create OAuth Client ID

Select Desktop App

Download the JSON file

Rename to:

client_secrets.json


Place it in the project folder (same directory as the Streamlit script)

▶️ Run the App
streamlit run streamlit_rag_gdrive.py

⚙️ Configuration (Sidebar)

Mistral API Key

Chunk Size

Overlap Size

Top-K Retrieval

Google Drive Folder ID

Buttons:

Fetch Documents from Google Drive

Ingest and Build Index

Run Query

❓ How It Works
1. Authenticate with Google Drive

User authorizes access → documents are fetched.

2. Ingest Documents

Text is chunked

Embeddings are computed

FAISS index is built

3. Query

Query is embedded

Top-K chunks retrieved

App builds a structured prompt

Mistral answers using retrieved context

📌 Example Query Flow

User types a question

System finds most relevant document chunks

Chunks are displayed

Mistral model generates an answer

Answer shown along with retrieved sources

🧱 Limitations

PyDrive PDF extraction is basic

No vector store persistence (memory only)

Do not upload sensitive data

Requires manual Google OAuth

🗺️ Future Improvements (Optional)

🔄 Persistent FAISS index

🧾 Better PDF parsing (PyMuPDF)

🧠 Use Mistral embedding models

🔐 Service account support

🧩 Add multi-page Streamlit

☁️ Deploy on Streamlit Cloud or GCP

🤝 Contributions

Pull requests are welcome!
If you want a feature added, feel free to open an issue.

📜 License

MIT License (or update to your preferred license)

If you want, I can also prepare:

📦 requirements.txt
📄 LICENSE file
📌 GitHub repo description
🖼️ README images/badges
🚀 Deployment instructions (Streamlit Cloud / Docker / GCP)
