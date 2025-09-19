import os
import streamlit as st
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from mistralai import Mistral

# =====================
# Google Drive Auth Helper
# =====================
def authenticate_gdrive():
    if not os.path.exists("client_secrets.json"):
        st.error("⚠️ Missing `client_secrets.json` file!\n\n**Fix:**\n1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials).\n2. Create OAuth client credentials (Desktop App).\n3. Download the JSON file and rename it to `client_secrets.json`.\n4. Place it in the same folder where you run this Streamlit app.\n\nThen restart the app and try again.")
        return None

    try:
        gauth = GoogleAuth()
        gauth.LoadClientConfigFile("client_secrets.json")
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
        st.success("✅ Google Drive authenticated successfully.")
        return drive
    except Exception as e:
        st.error(f"Google Drive authentication failed: {e}")
        return None

# =====================
# Document Fetching
# =====================
def fetch_drive_docs(drive, folder_id, min_docs=4):
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    if len(file_list) < min_docs:
        st.error(f"Found only {len(file_list)} documents. Please ensure there are at least {min_docs} in the folder.")
        return []

    docs = []
    for file in file_list:
        if file['mimeType'] == 'application/vnd.google-apps.document':
            content = file.GetContentString()
            docs.append({"title": file['title'], "content": content})
    return docs

# =====================
# Embedding + FAISS Index
# =====================
def build_faiss_index(docs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc['content'] for doc in docs]
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, model, docs

# =====================
# Mistral Query
# =====================
def query_mistral(index, model, docs, user_query, mistral_api_key):
    query_embedding = model.encode([user_query])
    D, I = index.search(query_embedding, k=3)
    retrieved_docs = [docs[i]['content'] for i in I[0]]

    context = "\n\n".join(retrieved_docs)
    prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    client = Mistral(api_key=mistral_api_key)
    resp = client.chat.complete(
        model="mistral-tiny",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message['content'], retrieved_docs

# =====================
# Streamlit UI
# =====================
st.title("📂 Domain-Specific RAG - Google Drive + MistralAI")

st.sidebar.header("🔑 Setup")
folder_id = st.sidebar.text_input("Google Drive Folder ID")
mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")

if st.sidebar.button("Fetch & Build Index"):
    drive = authenticate_gdrive()
    if drive:
        docs = fetch_drive_docs(drive, folder_id)
        if docs:
            index, model, stored_docs = build_faiss_index(docs)
            with open("faiss_index.pkl", "wb") as f:
                pickle.dump((index, model, stored_docs), f)
            st.success(f"✅ Indexed {len(docs)} documents successfully!")

if os.path.exists("faiss_index.pkl"):
    with open("faiss_index.pkl", "rb") as f:
        index, model, stored_docs = pickle.load(f)

    user_query = st.text_input("Ask a question about the documents:")
    if st.button("Run Query"):
        if not mistral_api_key:
            st.error("Please provide your Mistral API Key in the sidebar.")
        else:
            answer, context_docs = query_mistral(index, model, stored_docs, user_query, mistral_api_key)
            st.markdown("### 🔍 Retrieved Context")
            for i, doc in enumerate(context_docs):
                st.markdown(f"**Doc {i+1}:** {doc[:300]}...")
            st.markdown("### 🤖 Answer")
            st.success(answer)