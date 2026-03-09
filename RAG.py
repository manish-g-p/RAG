import streamlit as st
import os
from typing import Optional
from io import BytesIO
import google.genai as genai
from google.genai.errors import APIError

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


# Extract content from PDF and DOCX files
def read_document_content(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    try:
        if file_extension in ['.txt', '.md']:
            return uploaded_file.read().decode('utf-8')

        elif file_extension == '.pdf':
            if not PdfReader:
                return "Error: Cannot read pdf. Please install pypdf"
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

        elif file_extension == '.docx':
            if not Document:
                return "Error: Cannot read docx. Please install python-docx"
            doc = Document(BytesIO(uploaded_file.getvalue()))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text

        else:
            return f"Unsupported file type: {file_extension}"

    except Exception as e:
        return f"Error reading file: {str(e)}"


# Configuration for RAG
# Use environment variable to avoid hardcoding the API key
# Set BYTEZ_API_KEY in your environment or .env file

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env loading if python-dotenv not available
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

API_KEY = os.getenv("BYTEZ_API_KEY")
if not API_KEY:
    st.error(
        "API key not found. Please set the 'BYTEZ_API_KEY' "
        "environment variable or create a .env file.\n\n"
        "Example (Windows CMD): set BYTEZ_API_KEY=your-api-key-here\n"
        "Example (Windows PowerShell): $env:BYTEZ_API_KEY='your-api-key-here'\n"
        "Example (Linux/Mac): export BYTEZ_API_KEY='your-api-key-here'\n\n"
        "Or create a .env file with: BYTEZ_API_KEY=your-api-key-here"
    )
    st.stop()
MODEL_NAME = "google/gemini-2.5-flash"


class GeminiAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or API_KEY
        from bytez import Bytez
        self.sdk = Bytez(self.api_key)
        self.model = self.sdk.model(MODEL_NAME)

    def generate_content(self, model: str, contents: list, system_instruction: str) -> str:
        try:
            # Convert contents to bytez format with system instruction
            messages = [{"role": "system", "content": system_instruction}]
            for item in contents:
                if isinstance(item, dict) and "parts" in item:
                    # Handle Google GenAI format: {"parts": [{"text": "..."}]}
                    text = item["parts"][0].get("text", "")
                    messages.append({"role": "user", "content": text})
                elif isinstance(item, dict):
                    messages.append(item)
                elif isinstance(item, str):
                    messages.append({"role": "user", "content": item})
            
            response = self.model.run(messages)
            if response.error:
                return f"Error generating content: {response.error}"
            return response.output.get('content', str(response.output))
        except Exception as e:
            return f"Unexpected error: {str(e)}"


# Streamlit app
st.set_page_config(page_title="RAG with Gemini", layout="wide")
st.title("RAG with Gemini 2.5 Pro")
st.markdown("Upload a document and ask questions based on its content using Gemini 2.5 Pro.")

# Initialize session state
if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""
if "rag_response" not in st.session_state:
    st.session_state.rag_response = {}
if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = ""

# File uploader
uploaded_file = st.file_uploader("Upload a document (txt, md, pdf, docx)", type=["txt", "md", "pdf", "docx"])
if uploaded_file is not None:
    file_contents = read_document_content(uploaded_file)

    if file_contents.startswith("Error"):
        st.error(file_contents)
        st.session_state.uploaded_text = ""
        st.stop()
    else:
        st.session_state.uploaded_text = file_contents
        st.success("File uploaded and content extracted successfully!")

        with st.expander("Document Content"):
            display_text = file_contents[:2000]
            if len(file_contents) > 2000:
                display_text += '\n[...Truncated for display...]'
            st.code(display_text, language="text")

if not st.session_state.uploaded_text:
    st.info("Please upload a document to extract content and ask questions.")
    st.stop()

# User input for RAG
st.subheader("Ask Questions Based on the Document")
st.text_area("Enter your question here:", value=st.session_state.user_prompt_input,
             key="user_prompt_input", height=100)

gemini_api = GeminiAPI(api_key=API_KEY)


def run_rag_query():
    current_prompt = st.session_state.get('user_prompt_input', '').strip()
    if not current_prompt:
        st.warning("Please enter a question to ask.")
        return
    if not st.session_state.uploaded_text:
        st.warning("No document content available. Please upload a document first.")
        return

    st.session_state.rag_response = {'prompt': current_prompt, 'answer': None}
    with st.spinner(f"Augmenting generation for: '{current_prompt[:50]}...'"):
        system_instruction = "You are a helpful assistant that answers questions based on the provided document content."
        contents_payload = [
            {"parts": [{"text": st.session_state.uploaded_text}]},
            {"parts": [{"text": current_prompt}]}
        ]
        response_text = gemini_api.generate_content(
            model=MODEL_NAME,
            contents=contents_payload,
            system_instruction=system_instruction
        )
        st.session_state.rag_response['answer'] = response_text


st.button("Get grounded answer", on_click=run_rag_query, type="primary")

# Output box for the response
st.subheader("RAG response")
if st.session_state.rag_response.get('answer'):
    st.markdown(f"**Question:** {st.session_state.rag_response['prompt']}")
    st.markdown(f"**Answer:** {st.session_state.rag_response['answer']}")
else:
    st.info("Your answer will appear here once you ask a question and get a response from Gemini.")

st.markdown("---")
st.caption("Note: The quality of the response may vary based on the content of the document and the nature of the question asked. For best results, ensure that the document is well-structured and contains relevant information related to your question.")
