# RAG (Retrieval-Augmented Generation) Application Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [Voice Interaction](#voice-interaction)
7. [Usage](#usage)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)
11. [License](#license)

## Introduction

The RAG (Retrieval-Augmented Generation) application is a powerful tool that combines the capabilities of large language models with efficient document retrieval. It allows users to upload various document types (PDF, TXT, CSV, Excel) and ask questions about their content, receiving accurate and contextually relevant answers. The application now features voice interaction capabilities, allowing for hands-free operation and accessibility.

## System Architecture

### High-Level Overview

```
┌─────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Document Input │───▶│  Document Loading  │───▶│  Text Splitting   │
└─────────────────┘    └───────────────────┘    └────────┬──────────┘
                                                         │
┌─────────────────┐    ┌───────────────────┐    ┌───────▼──────────┐
│  User Question  │    │  Query Processing │    │  Vector Store    │
│  (Text/Voice)   │───▶│  & Rewriting      │◀───┤  (FAISS)         │
└─────────────────┘    └────────┬──────────┘    └────────┬──────────┘
                                │                        │
                        ┌───────▼────────────────────────▼──────────┐
                        │          LLM (OpenAI/Ollama)              │
                        │          Context Integration              │
                        └───────────────────┬────────────────────────┘
                                            │
                                    ┌───────▼──────────┐    ┌────────────────┐
                                    │  Response        │    │  Voice Output  │
                                    │  Generation      │───▶│  (Text-to-Speech) │
                                    └──────────────────┘    └────────────────┘
```

### Components

1. **Document Processing**
   - Handles multiple file formats (PDF, TXT, CSV, Excel)
   - Text extraction and cleaning
   - Chunking of documents for efficient retrieval

2. **Voice Interaction**
   - Voice input for questions (speech-to-text)
   - Text-to-speech for responses
   - Real-time voice control
   - Stop functionality for voice output

2. **Vector Store**
   - FAISS-based vector database for efficient similarity search
   - Caching mechanism for improved performance
   - Support for both CPU and GPU acceleration

3. **Language Models**
   - Integration with OpenAI models
   - Support for local Ollama models
   - Query rewriting for improved retrieval

4. **User Interface**
   - Gradio-based web interface
   - Support for file uploads
   - Interactive chat interface

## Voice Interaction

The application includes comprehensive voice interaction capabilities:

### Voice Input
- Click the 🎤 button to start voice recording
- Speak your question clearly into the microphone
- The application will convert your speech to text automatically
- Press Enter or click Send to submit your question

### Voice Output
- Responses can be read aloud using text-to-speech
- Toggle voice output on/off using the "Enable Voice Output" checkbox
- Click the "Stop" button to immediately stop voice output

### Voice Controls
- **Start Recording**: Click the microphone button
- **Stop Recording**: Click the microphone button again or wait 5 seconds
- **Stop Speech**: Click the red "Stop" button to halt voice output
- **Toggle Voice Output**: Use the checkbox to enable/disable text-to-speech

### Requirements
- Microphone for voice input
- Speakers or headphones for voice output
- Internet connection for speech recognition (Google's Web Speech API)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Tesseract OCR (for PDF image extraction)
- Poppler (for PDF processing)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RAG_APP
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key  # If using OpenAI models
   TESSERACT_CMD=/path/to/tesseract  # Optional, for OCR
   POPPLER_PATH=/path/to/poppler/bin  # Optional, for PDF processing
   ```

## Configuration

The application can be configured using environment variables:

- `OPENAI_API_KEY`: API key for OpenAI services (required if using OpenAI models)
- `OLLAMA_API_BASE`: Base URL for Ollama API (default: http://localhost:11434)
- `OLLAMA_MODEL`: Default Ollama model to use (default: llama2)
- `TESSERACT_CMD`: Path to Tesseract OCR executable
- `POPPLER_PATH`: Path to Poppler binaries (for PDF processing)

## Core Components

### Document Processing

The application supports multiple document types:

1. **PDF Documents**
   - Extracts text using PyPDF
   - Falls back to OCR for image-based PDFs
   - Preserves document structure and formatting

2. **Text Files**
   - Handles plain text files
   - Supports various encodings
   - Preserves line breaks and formatting

3. **CSV/Excel Files**
   - Converts tabular data into structured documents
   - Preserves column headers and data types
   - Handles large datasets efficiently

### Vector Store

- Uses FAISS for efficient similarity search
- Implements caching for improved performance
- Supports both CPU and GPU acceleration
- Handles large document collections

### Language Models

#### Supported Models
1. **OpenAI Models**
   - Requires API key
   - High-quality responses
   - Various model sizes available

2. **Ollama Models**
   - Local model support
   - Privacy-focused
   - Custom model support

## Usage

### Starting the Application

```bash
python ui_app.py
```

The application will start a local web server (default: http://localhost:7860)

### Using the Web Interface

1. **Upload a Document**
   - Click "Upload" to select a file (PDF, TXT, CSV, Excel)
   - Wait for the document to be processed

2. **Ask Questions**
   - Type your question in the chat interface
   - The system will retrieve relevant information and generate an answer

3. **View Results**
   - The conversation history is displayed in the chat window
   - Source documents are cited for reference

### Command Line Interface

The application can also be used programmatically:

```python
from rag_core import build_vector_store_for_file, build_context_and_answer

# Build vector store for a document
vector_store, _ = build_vector_store_for_file("document.pdf")

# Get answer to a question
response = build_context_and_answer(
    vector_store=vector_store,
    question="What is the main topic of this document?",
    openai_model=openai_model,
    openai_model_name="gpt-4",
    ollama_model=None,
    ollama_model_name=None
)

print(response["answer"])
```

## API Reference

### Core Functions

#### `build_vector_store_for_file(file_path: str) -> tuple[FAISS, str]`

Builds a FAISS vector store from a document file.

**Parameters:**
- `file_path`: Path to the input file (PDF, TXT, CSV, Excel)

**Returns:**
- Tuple containing the FAISS vector store and a status message

#### `build_context_and_answer(vector_store: FAISS, question: str, **kwargs) -> dict`

Generates an answer to a question using the provided vector store.

**Parameters:**
- `vector_store`: FAISS vector store with document embeddings
- `question`: User's question
- `openai_model`: Optional OpenAI model instance
- `ollama_model`: Optional Ollama model instance
- `history`: Optional conversation history

**Returns:**
- Dictionary containing the answer and metadata

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all required packages are installed
   - Check that Tesseract and Poppler are properly installed and in PATH

2. **API Key Issues**
   - Verify that your OpenAI API key is set in the `.env` file
   - Ensure the key has sufficient permissions

3. **Memory Errors**
   - Reduce the batch size in `rag_core.py` for large documents
   - Use a smaller embedding model if running out of GPU memory

4. **OCR Failures**
   - Ensure Tesseract is properly installed
   - Check that the language data for Tesseract is available

## Performance Considerations

### Hardware Requirements

- **CPU**: Modern multi-core processor recommended
- **RAM**: Minimum 8GB, 16GB+ recommended for large documents
- **GPU**: Optional but recommended for faster processing

### Optimization Tips

1. **Document Size**
   - Split large documents into smaller chunks
   - Use appropriate chunk sizes (default: 1000 tokens)

2. **Model Selection**
   - Use smaller models for faster inference
   - Consider quantized models for CPU-only environments

3. **Caching**
   - The application caches vector stores for improved performance
   - Clear the cache directory if disk space is a concern

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on the project repository.

---

*Documentation generated on December 6, 2024*
