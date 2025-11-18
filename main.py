import os
import pyttsx3 

# Loaders, Embeddings, Vectorstores, and LLMs are in 'langchain-community'
from langchain_community.document_loaders import TextLoader 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
# Text splitters are in 'langchain-text-splitters'
from langchain_text_splitters import CharacterTextSplitter 
# The legacy RetrievalQA chain is found in 'langchain-classic'
from langchain_classic.chains import RetrievalQA

# Configuration
DATA_FILE = "speech.txt"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
tts_engine = None 

#  TTS Engine Initialization 
try:
    tts_engine = pyttsx3.init()
    # Adjust speed for clearer speech
    rate = tts_engine.getProperty('rate')
    tts_engine.setProperty('rate', rate - 50) 
except Exception as e:
    # if audio fails then print
    print(f"Warning: TTS engine initialization failed ({e}). Proceeding without audio.")
    tts_engine = None

def speak(text_to_speak):
    """Converts the given text string to speech using pyttsx3."""
    if tts_engine:
        tts_engine.say(text_to_speak)
        tts_engine.runAndWait()

def load_and_split_data():
    """1. Load the text, and 2. split it into chunks."""
    print("1. Loading document and splitting...")
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    # Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    return texts

def setup_vector_store(texts):
    """3. Create Embeddings and store them in ChromaDB."""
    print(f"2. Initializing embeddings with {EMBEDDING_MODEL}...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create the vector store from documents and save it locally
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH 
    )
    vectordb.persist()
    print(f"-> Vector database built and saved to '{CHROMA_DB_PATH}'. (Step 3 complete)")
    
    return vectordb

def load_existing_vector_store():
    """Loads the vector store if it already exists."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH, 
        embedding_function=embeddings
    )
    return vectordb

def setup_qa_chain(vectordb):
    """4. Retrieve relevant chunks and 5. Generate an answer."""
    print(f"3. Initializing LLM: Ollama with {LLM_MODEL}...")
    
    llm = Ollama(model=LLM_MODEL) 
    
    # Create a retriever interface
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Orchestrate the RAG pipeline using RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )
    print("-> RAG Chain setup complete.")
    
    return qa_chain

def main():
    # Data Setup (Steps 1, 2, 3)
    if not os.path.exists(CHROMA_DB_PATH):
        print("Starting data processing and vector store creation (first run)...")
        texts = load_and_split_data()
        vectordb = setup_vector_store(texts)
    else:
        print("Loading existing vector store...")
        vectordb = load_existing_vector_store()

    # QA System Setup (Steps 4, 5)
    qa_system = setup_qa_chain(vectordb)

    # Command Line Interface
    print("\n--- AmbedkarGPT CLI Ready ---")
    if tts_engine:
        print("Audio output is enabled.")
    print("Ask a question based on the speech text (type 'exit' to quit).")
    
    while True:
        question = input("Your Question: ")
        
        if question.lower() == 'exit':
            print("Exiting Q&A system. Goodbye!")
            speak("Goodbye!")
            break
            
        if not question.strip():
            continue

        try:
            result = qa_system({"query": question})
            final_answer = result["result"].strip()
            
            print("\n**Answer:**", final_answer)
            
            # Speak the answer aloud
            speak(final_answer) 
            
            # Print the source documents for verification
            print("\n**Context Retrieved (Source Documents):**")
            for i, doc in enumerate(result["source_documents"]):
                print(f"  [{i+1}] Source: '{doc.page_content.strip()}'")
            print("-" * 20)

        except Exception as e:
            print(f"\n--- ERROR ---\nAn error occurred: {e}")
            print("Troubleshooting: Ensure Ollama is running and 'mistral' model is pulled.")

if __name__ == "__main__":
    main()