# Agentic-RAG-Chatbot
This project is an **Agent-Based Retrieval-Augmented Generation (RAG) chatbot** designed with modular agents communicating asynchronously using a custom message-passing protocol called **Model Context Protocol (MCP)**. It allows users to upload documents, ask context-aware questions, and get LLM-generated responses based on semantic search from uploaded files.


## Features

- Upload multi-format documents: `.pdf`, `.docx`, `.pptx`, `.csv`, `.txt`, `.md`
- Automatic document parsing and intelligent text chunking
- FAISS-based semantic search over document chunks
- LLM-powered contextual answering
- Asynchronous message passing via MCP bus
- Modular agent architecture (Ingestion, Retrieval, Response)
- Streamlit-based web interface


## Agent Architecture

sequenceDiagram
    UI->>IngestionAgent: UPLOAD
    IngestionAgent->>RetrievalAgent: DOCUMENTS_PARSED
    UI->>RetrievalAgent: QUERY
    RetrievalAgent->>LLMResponseAgent: CONTEXT_RESPONSE
    LLMResponseAgent->>UI: Answer
