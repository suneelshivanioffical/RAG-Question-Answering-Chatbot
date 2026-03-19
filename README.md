# RAG-Question-Answering-Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF/DOCX files and ask questions about them using Google Gemini AI.

The app retrieves relevant document context and generates accurate answers with conversational memory.

## Demo

![]()


## Overview

- Streamlit provides an interactive web interface for users
- Google Gemini API generates intelligent, human-like responses
- LangChain manages the workflow between document processing and the AI model
- FAISS as the vector database for fast similarity search enables fast retrieval.
- HuggingFace Embeddings converts text into numerical vectors for semantic understanding

## Project Structure

```
RAG-Question-Answering-Chatbot
│
├── app.py               # Streamlit application
├── requirements.txt     # Project dependencies
└── README.md            # Documentation
```

## Project Features

- Upload and process PDF & DOCX files
- Ask questions from your documents
- Context-aware answers using RAG
- Fast retrieval with FAISS vector search
- Supports follow-up questions (chat memory)
- Secure Gemini API key integration
- Simple and interactive Streamlit UI

## Technologies

- Python
- Streamlit
- Gemini API (google-genai)
- LangChain 
- FAISS (Vector Database)
- Sentence Transformers (Embeddings)

## Model

Embedding Model

    ```
    all-MiniLM-L6-v2
    ```
Used to convert text into vector embeddings for semantic search.

LLM (Language Model)

    ```
    gemini-2.5-flash
    ```

Used for generating responses based on retrieved document context.

## Run the Project Locally

1. Create a Conda Environment

    ```bash
    conda create -n rag_chatbot python=3.10
    conda activate rag_chatbot
    ```

2. Install requirements for run
   
    ```bash
    pip install -r requirements.txt
    ```
3. Start the Server
   
    ```bash
    streamlit run app.py
    ```
After running the command, the Streamlit web application will open in your browser.
