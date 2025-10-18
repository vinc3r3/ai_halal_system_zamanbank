from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pypdf import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Configuration ---
# Define the PDF file path and the directory to store the Chroma database
PDF_FILE_PATH = "data_pdfs/aifc-islamic-banking-business-prudential-rules.pdf"
CHROMA_DB_PATH = "./rag_knowledge_base"

# Define the open-source embedding model to use
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

# --- RAG Component Functions ---

def load_pdf_data(file_path):
    """
    Reads the text content from a PDF file.
    
    Args:
        file_path (str): The path to the PDF document.
        
    Returns:
        str: The combined text content of the PDF.
    """
    print(f"Loading document from: {file_path}...")
    try:
        reader = PdfReader(file_path)
        # Extract text from all pages
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None

def split_text_into_chunks(text, chunk_size=1500, chunk_overlap=150):
    """
    Splits the document text into smaller, overlapping chunks.
    
    Args:
        text (str): The full text of the document.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The overlap between consecutive chunks.
        
    Returns:
        list: A list of LangChain Document objects.
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Create LangChain Document objects from the raw text
    chunks = text_splitter.create_documents([text])
    print(f"Text successfully split into {len(chunks)} chunks.")
    return chunks

def add_metadata_to_chunks(chunks, base_metadata):
    """
    Applies custom metadata to each chunk.
    
    Args:
        chunks (list): A list of LangChain Document objects.
        base_metadata (dict): The custom metadata to apply to all chunks from this document.
        
    Returns:
        list: The list of chunks with updated metadata.
    """
    # Define a default source based on the PDF path, assuming the file path is the source
    source_path = base_metadata.get("source", os.path.basename(PDF_FILE_PATH))
    
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            **base_metadata, # Unpack the provided metadata
            "chunk_index": i + 1,
            "source": source_path # Ensure the source is correctly set
        }
    return chunks

def create_and_save_vector_db(chunks_with_metadata, db_path, model_name):
    """
    Generates embeddings and saves the chunks into a Chroma vector database 
    using a local Hugging Face model.
    
    Args:
        chunks_with_metadata (list): A list of LangChain Document objects with metadata.
        db_path (str): The file path to save the persistent Chroma database.
        model_name (str): The name of the Sentence Transformer model to use.
    """
    print(f"Loading embedding model: {model_name}...")
    
    # 1. Initialize the HuggingFace embedding model
    # The model will be automatically downloaded the first time it's run
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("Generating embeddings and saving to Chroma DB...")
    
    # 2. Create the Chroma vector store from the documents
    vectorstore = Chroma.from_documents(
        documents=chunks_with_metadata,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    # 3. Persist the database to disk
    # This automatically happens in the __del__ method of the Chroma class 
    # but we can call it explicitly for safety.
    vectorstore.persist()
    print(f"Knowledge Base successfully created and saved to: {db_path}")

# --- Main Execution ---

if __name__ == "__main__":
    
    # --- Custom Metadata Definition ---
    # Define the custom metadata for the document being processed
    CUSTOM_METADATA = {
        "some_id": "AIFC-PRUDENTIAL-RULES-2023", 
        "type": "sheikh_interpretation/regulatory_document", 
        "general_question/topic": "Prudential regulation for Islamic banks in AIFC",
        "answer/explanation": "This document outlines the capital adequacy, risk management, and disclosure requirements for Islamic banking institutions operating under the Astana International Financial Centre (AIFC) framework."
    }
    
    # 1. Load PDF Data (using the file path from your previous attempt)
    document_text = load_pdf_data(PDF_FILE_PATH)

    if document_text:
        # 2. Split Text into Chunks
        chunks = split_text_into_chunks(document_text, chunk_size=1500, chunk_overlap=150)
        
        # 3. Add Custom Metadata
        chunks_with_metadata = add_metadata_to_chunks(chunks, CUSTOM_METADATA)
        
        # 4. Create and Save Vector Database
        create_and_save_vector_db(
            chunks_with_metadata, 
            CHROMA_DB_PATH,
            EMBEDDING_MODEL_NAME
        )