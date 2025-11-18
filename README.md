# AmbedkarGPT-Intern-Task
Local RAG Q&amp;A system built with LangChain, Ollama (Mistral 7B), and ChromaDB. It answers questions based solely on the source text (speech.txt). The system uses a 100% free, open-source stack and features a Text-to-Speech (TTS) enhancement for interactive audio output.

#################################################

# AmbedkarGPT-Intern-Task

This repository contains the solution for an AI Project. It implements a command-line **Retrieval-Augmented Generation (RAG) Q&A system** designed to answer questions based **solely** on the content of the provided `speech.txt` file. The entire system is built on **free, open-source, and local technologies**.

##  Technical Architecture Overview

The pipeline is orchestrated by the modular **LangChain** framework and adheres to a strict set of technical constraints.

### 1. The RAG Pipeline Components

The Q&A system uses the following components, demonstrating a 100% local, no-cost stack:

| RAG Step | Technology Used | Implementation Detail | Constraint Alignment |

| **LLM / Generator** | **Ollama** with **Mistral 7B** | Local server and model for text generation. | 100% free; no API keys. |
| **Embeddings** | **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`) | Local model for converting text chunks into semantic vectors. | Local, no API keys. |
| **Vector DB / Retriever** | **ChromaDB** | Persistent local vector store (`chroma_db`) for efficient similarity search. | Local, open-source. |
| **Orchestration** | **LangChain** (Modular) | Uses `RetrievalQA` to chain retrieval (ChromaDB) and generation (Ollama). | Core framework utilized. |
| **Enhancement** | **pyttsx3** | Added Text-to-Speech (TTS) feature for enhanced user output (runs on native OS voice drivers). | Functional enhancement. |

### 2. Execution Flow

1.  **Preparation:** `speech.txt` is loaded, split via `CharacterTextSplitter`, and vectorized using the local embedding model.
2.  **Storage:** The resulting vectors are persisted in **ChromaDB**.
3.  **Query:** A user's question is passed to the **LangChain RetrievalQA** chain.
4.  **Retrieval:** The query is vectorized and searched against the ChromaDB, returning only the most relevant text chunks from the source document.
5.  **Generation:** These retrieved chunks, along with the user's question, are passed to the **Ollama/Mistral 7B** model, forcing it to generate an answer based **solely** on the provided context.

## ⚙️ Quick Start

Assuming Python 3.8+ and **Ollama** are installed:

1.  **Model Download:** `ollama pull mistral`
2.  **Setup Environment:** Create and activate a Conda environment.
3.  **Install Dependencies:** `pip install -r requirements.txt`
4.  **Run:** Ensure Ollama is running, and execute `python main.py` in the project directory.
