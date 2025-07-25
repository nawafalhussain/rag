import streamlit as st
from api_utils import upload_document, list_documents, delete_document
import requests





def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            # Each model has a 'name' field
            models = [m['name'] for m in data.get('models', [])]
            return models
        else:
            st.sidebar.error("Failed to fetch models from Ollama")
            return []
    except Exception as e:
        st.sidebar.error(f"Error contacting Ollama: {str(e)}")
        return []
    

def display_sidebar():
    # Model selection
    model_options = ["gpt-4o", "gpt-4o-mini"]
    model_options = get_ollama_models()
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    # Document upload
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "html"])
    if uploaded_file and st.sidebar.button("Upload"):
        with st.spinner("Uploading..."):
            upload_response = upload_document(uploaded_file)
            if upload_response:
                st.sidebar.success(f"File uploaded successfully with ID {upload_response['file_id']}.")
                st.session_state.documents = list_documents()

    # List and delete documents
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        st.session_state.documents = list_documents()

    # Display document list and delete functionality
    if "documents" in st.session_state and st.session_state.documents:
        for doc in st.session_state.documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']})")

        selected_file_id = st.sidebar.selectbox("Select a document to delete", 
                                                options=[doc['id'] for doc in st.session_state.documents])
        if st.sidebar.button("Delete Selected Document"):
            delete_response = delete_document(selected_file_id)
            if delete_response:
                st.sidebar.success(f"Document deleted successfully.")
                st.session_state.documents = list_documents()
