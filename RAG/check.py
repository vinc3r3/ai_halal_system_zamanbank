import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DB_PATH = "./rag_knowledge_base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

def inspect_vector_db(db_path, model_name):
    """
    Loads the persistent ChromaDB and checks its contents.
    """
    print(f"--- Loading existing Knowledge Base from: {db_path} ---")
    
    # 1. Initialize the embedding function (MUST match the one used for creation)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # 2. Load the persistent vector store
    try:
        # Note: We re-initialize the Chroma object, which loads the existing data
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    # 3. Get the underlying Chroma Client collection
    # LangChain's Chroma wrapper provides access to the raw Chroma API via ._collection
    collection = vectorstore._collection
    
    # --- Verification Step 1: Count ---
    count = collection.count()
    print(f"Total documents found in the database: {count}")

    # --- Verification Step 2: Retrieve a Sample (e.g., the first 5 documents) ---
    print("\n--- Retrieving the first 5 documents for verification ---")
    
    # Use the Chroma client's .get() method to fetch documents by offset
    results = collection.get(
        limit=5,          # Retrieve the first 5
        offset=0,         # Start from the beginning
        include=['metadatas', 'documents'] # Specify which data to return
    )

    if not results['documents']:
        print("Database is empty or retrieval failed.")
        return

    # 4. Print the results clearly
    for i in range(len(results['documents'])):
        print(f"\n--- Document {i+1} ---")
        
        # Display the custom metadata
        print("METADATA:")
        for k, v in results['metadatas'][i].items():
            print(f"  {k}: {v}")
            
        # Display the chunk content
        print("\nCONTENT (Chunk Text):")
        # Use a slice to show only the first 500 characters for brevity
        content = results['documents'][i]
        print(f"  {content[:500]}...")
        if len(content) > 500:
             print(f"  [... {len(content) - 500} more characters ...]")


# --- Execution ---
if __name__ == "__main__":
    inspect_vector_db(CHROMA_DB_PATH, EMBEDDING_MODEL_NAME)