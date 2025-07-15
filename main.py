from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import os
import tempfile
import requests
from pdf2image import convert_from_path
import pytesseract
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ====== Configuration ======
UPLOAD_DIR = "uploaded_files"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ====== FastAPI Init ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev. Restrict for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== In-memory metadata store for listing/deleting ======
documents_db = {}

# ====== Chroma Vector DB Setup ======
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_fn)

# ====== Pydantic Models ======
class ChatRequest(BaseModel):
    question: str
    model: str
    session_id: str | None = None

# ====== /upload-doc Endpoint ======
@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    # Save file to disk
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as out_file:
        content = await file.read()
        out_file.write(content)

    # Use temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.filename.split(".")[-1]) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Extract text
    if file.filename.endswith(".pdf"):
        all_text = ""
        images = convert_from_path(tmp_path, dpi=300)
        for img in images:
            all_text += pytesseract.image_to_string(img, lang='ara') + "\n"
    elif file.filename.endswith(".docx"):
        from docx import Document as DocxDocument
        doc = DocxDocument(tmp_path)
        all_text = "\n".join([p.text for p in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_text(all_text)
    metadatas = [{"filename": file.filename, "chunk_id": i} for i in range(len(splits))]
    chroma_db.add_texts(splits, metadatas=metadatas)
    chroma_db.persist()

    # Store metadata for listing/deleting
    file_id = str(uuid4())
    documents_db[file_id] = {
        "filename": file.filename,
        "filepath": save_path,
        "chunks": len(splits)
    }
    return {"file_id": file_id, "filename": file.filename}

# ====== /list-docs Endpoint ======
@app.get("/list-docs")
async def list_docs():
    return [
        {"id": file_id, "filename": doc["filename"], "chunks": doc["chunks"]}
        for file_id, doc in documents_db.items()
    ]

# ====== /delete-doc Endpoint ======
@app.post("/delete-doc")
async def delete_doc(file_id: str):
    if file_id not in documents_db:
        raise HTTPException(status_code=404, detail="File not found")
    filename = documents_db[file_id]["filename"]
    # Remove from Chroma
    # NOTE: Chroma does not support deleting by metadata directly in the open-source version.
    # So for demo, we will just remove metadata and (optionally) file from disk.
    del documents_db[file_id]
    # Optionally, remove file from disk
    # os.remove(documents_db[file_id]["filepath"])
    return {"deleted": file_id, "filename": filename}



def call_ollama(model, prompt):
    ollama_url = "http://localhost:11434/api/generate"
    ollama_payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        response = requests.post(ollama_url, json=ollama_payload, stream=True, timeout=120)
        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    obj = json.loads(line.decode('utf-8'))
                    answer += obj.get("response", "")
                except Exception as e:
                    continue
        return answer
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

# ====== /chat Endpoint ======
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Retrieve top relevant chunks
    top_k = 4
    search_results = chroma_db.similarity_search(request.question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in search_results])

    # 2. Build prompt for LLM
    prompt = (
        f"Answer the following question based only on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {request.question}\n"
        f"Answer:"
    )
    answer = call_ollama(request.model, prompt)
    for doc in search_results:
        print("DEBUG:", doc.metadata)
    return {
    "answer": answer,
    "model": request.model,
    "session_id": request.session_id or str(uuid4()),
    "retrieved_chunks": [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("filename", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", -1)
        }
        for doc in search_results
    ]
}



# ====== Root Endpoint ======
@app.get("/")
def root():
    return {"msg": "FastAPI RAG backend is running!"}
