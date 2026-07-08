# 🧠 RAG Document Question Answering System

This project implements a highly interactive, customizable, and feature-rich **Retrieval-Augmented Generation (RAG)** system that allows users to upload custom documents (PDFs, plain text files, notes, resumes) and query them for semantic and grounded answers.

Instead of relying solely on the pre-trained weights of a Large Language Model (which can lead to hallucinations), this pipeline retrieves the most relevant context snippets from your uploaded files and uses them to ground the LLM's responses, ensuring accuracy and data privacy.

---

## 🚀 Key Features

*   **📂 Ingest Multiple Formats:** Supports uploading `.pdf` and `.txt` documents.
*   **🛠️ Adjustable RAG Tuning Sidebar:**
    *   **Chunk Size & Overlap:** Control how documents are parsed and split live.
    *   **Top-K Chunks:** Control how many matching sections are fed to the model.
    *   **System Prompt Customization:** Fine-tune model guidelines directly from the UI.
*   **🔋 Dual-Layer Vector Database (FAISS & NumPy Fallback):**
    *   Tries to use **FAISS** (Facebook AI Similarity Search) for rapid vector search.
    *   Includes a built-in, zero-dependency **NumPy-based Vector Engine** as a fallback. If compiled binaries for FAISS fail to install on Windows/macOS, the app runs seamlessly without crashing.
*   **⚙️ Multi-Engine Providers:**
    *   **Local / Offline Mode:** Completely free, using local embeddings (`all-MiniLM-L6-v2`) and a mock response generator (no internet or API keys needed).
    *   **Google Gemini API:** Fast, high-fidelity embeddings (`text-embedding-004`) and generation (`gemini-1.5-flash` or `gemini-1.5-pro`).
    *   **Cohere API:** Advanced multi-stage embeddings (`embed-english-v3.0`) and Chat connector (`command-r-plus`).
*   **🔍 Interactive Chunk Inspector:** View and slide through the individual parsed text chunks of your documents to see exactly how text-splitting works under the hood.
*   **📊 Diagnostics Telemetry:** Renders real-time execution metrics for text extraction, chunking, embedding generation, query retrieval, and answer synthesis.
*   **📚 Grounded Sources:** Every AI answer displays neat expander cards listing the source file name, chunk index, text preview, and vector similarity score.

---

## 📂 Project Directory Structure

```text
RAG_Document_Question_Answering/
├── app.py                # Main Streamlit UI, styling sheet, & diagnostic panel
├── rag_pipeline.py       # Document loaders, chunking logic, vector store, and API connectors
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (this file)
└── sample_docs/          # Example documents for immediate testing
    ├── clinical_faq.txt
    ├── rag_concept.txt
    └── clinical_workstation_research_report.pdf
```

---

## 🛠️ Installation & Setup

Follow these steps to run the RAG system locally:

### 1. Clone or Download the Directory
Navigate to the directory on your system:
```bash
cd path/to/RAG_Document_Question_Answering
```

### 2. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to prevent package version conflicts:
*   **Windows (Command Prompt):**
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```
*   **Windows (PowerShell):**
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
*   **macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies
Install all package requirements defined in the file:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Start the Streamlit local development web server:
```bash
streamlit run app.py
```

Streamlit will automatically launch the interface in your default web browser (usually at `http://localhost:8501`).

---

## 💡 How to Test the System

1.  **Select Provider:** On the sidebar, leave the setting on **Local / Offline Mode** to test the system with zero API keys. 
2.  **Upload Documents:** Expand the **Ingest Custom Documents** card in the main panel, click to select the sample files under the `sample_docs/` folder, and hit **⚡ Build Index**.
3.  **Inspect Chunks:** Open the **Document & Chunk Inspector** tab. Select a file and slide the chunk indicator to inspect how the recursive characters splitter broke the document into overlapping chunks.
4.  **Query:** Go to the Chat tab and type a question in the input box at the bottom, e.g., *"What is the NPI of the default clinician?"* or *"What is the advantage of using RAG?"*
5.  **View Grounding:** Notice the speed metric card update. Click **Grounded Context** under the response bubble to verify which exact text chunks were retrieved to answer your question.
6.  **Add API Key:** Paste your **Google Gemini** or **Cohere** API key in the sidebar, select the corresponding model, and ask more complex questions to see the LLM generate fluent responses based on your documents!

---

## 📄 License
This project is open-source and available under the MIT License.
