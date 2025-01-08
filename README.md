# Document-interaction-AI-agent

This project implements a document interaction AI agent that allows users to upload documents, ask questions, and generate literature reviews. It uses Ollama for language processing, RAG for contextual understanding, and ChromaDB for vector storage.

## Features

-   Upload PDF/DOCX documents.
-   Ask questions about uploaded documents.
-   Generate literature reviews.
-   Local Ollama models for language processing.
-   Retrieval Augmented Generation (RAG).
-   ChromaDB for efficient vector storage.
-   Web UI with Flask and Tailwind CSS.
-   Simple API.

## Technologies Used

-   Ollama
-   Langchain
-   ChromaDB
-   Flask
-   Tailwind CSS
-   Python
-   Sentence Transformers
-   PyPDFLoader/UnstructuredWordDocumentLoader

## Setup

1.  Clone: `git clone https://github.com/JimmyXtesla/Document-interaction-AI-agent/`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Install Ollama (see [ollama.com](https://ollama.com/)).
4.  Download an Ollama model (e.g., `ollama pull llama2`).
5.  Run: `python app.py`
6.  Open your browser to `http://127.0.0.1:5000/`.

## Usage

1.  Upload documents.
2.  Ask questions or enter review topic.
3.  Click on 'Ask Question' or 'Generate Review'.

## Contributing

Contributions welcome! Fork, branch, and submit a pull request.

## License

[MIT License]
