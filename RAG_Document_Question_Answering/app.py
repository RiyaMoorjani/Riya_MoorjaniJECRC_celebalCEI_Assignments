import os
import time
import streamlit as st
import numpy as np

# Import RAG pipeline backend components
from rag_pipeline import (
    extract_text,
    chunk_text,
    EmbeddingService,
    RAGVectorStore,
    generate_answer,
    HAS_FAISS,
    HAS_SENTENCE_TRANSFORMERS
)

# ==========================================
# 1. Page Configuration & Custom CSS Theme
# ==========================================
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# A sleek modern Dark/Neon-Glass theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');
    
    /* Apply modern fonts */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Outfit', sans-serif !important;
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
    }
    
    /* Clean, modern sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }
    
    /* Streamlit widgets modifications */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35) !important;
    }
    
    /* Card panel container */
    .telemetry-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 12px;
    }
    .telemetry-value {
        font-size: 20px;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'Space Grotesk', sans-serif;
    }
    .telemetry-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom source card wrapper */
    .source-card {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid #475569 !important;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
        border-left: 4px solid #6366f1 !important;
    }
    .source-header {
        font-size: 12px;
        font-weight: 600;
        color: #818cf8;
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    
    /* Chat message layouts */
    .chat-bubble {
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 12px;
        line-height: 1.5;
        font-size: 14.5px;
    }
    .chat-user {
        background-color: #1e293b;
        color: #e2e8f0;
        border-bottom-right-radius: 2px;
        margin-left: 20%;
        border: 1px solid #334155;
    }
    .chat-assistant {
        background: linear-gradient(180deg, #111c33 0%, #0d1527 100%);
        color: #f1f5f9;
        border-bottom-left-radius: 2px;
        margin-right: 20%;
        border: 1px solid #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. State Initialization
# ==========================================
if "documents" not in st.session_state:
    st.session_state.documents = {}  # filename -> {"text": str, "chunks": list, "char_count": int}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "telemetry" not in st.session_state:
    st.session_state.telemetry = {
        "ingest_time": 0.0,
        "chunk_count": 0,
        "embed_time": 0.0,
        "search_time": 0.0,
        "generate_time": 0.0
    }

# ==========================================
# 3. Sidebar Configuration Panel
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Configuration</h2>", unsafe_allow_html=True)

# Select Model Provider
st.sidebar.subheader("AI Model Provider")
provider = st.sidebar.selectbox(
    "Select API Provider",
    ["Local / Offline Mode", "Google Gemini API", "Cohere API"],
    index=0
)

# API Keys conditional input
api_key = ""
if provider == "Google Gemini API":
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Input your Gemini/Google AI Studio API Key. Will override any env variable."
    )
elif provider == "Cohere API":
    api_key = st.sidebar.text_input(
        "Cohere API Key",
        value=os.getenv("COHERE_API_KEY", ""),
        type="password",
        help="Input your Cohere client API Key. Will override any env variable."
    )

# Model configuration mappings
embedding_provider = "local"
llm_provider = "mock"
llm_model = None

if provider == "Google Gemini API":
    embedding_provider = "google"
    llm_provider = "google"
    llm_model = st.sidebar.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
elif provider == "Cohere API":
    embedding_provider = "cohere"
    llm_provider = "cohere"
    llm_model = st.sidebar.selectbox("Cohere Model", ["command-r-plus", "command-r"])
else:
    embedding_provider = "local"
    llm_provider = "mock"

# Embedding Dimension configuration based on choice
embedding_dim = 384  # Default for all-MiniLM-L6-v2 (local)
if embedding_provider == "google":
    embedding_dim = 768  # text-embedding-004
elif embedding_provider == "cohere":
    embedding_dim = 1024 # embed-english-v3.0

st.sidebar.divider()

# RAG Hyperparameters
st.sidebar.subheader("RAG Parameters")
chunk_size = st.sidebar.slider(
    "Chunk Size (Characters)",
    min_value=100,
    max_value=2000,
    value=500,
    step=50,
    help="Size of text chunks that documents are broken into."
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap (Characters)",
    min_value=0,
    max_value=500,
    value=100,
    step=10,
    help="Character overlap between successive chunks to preserve context."
)

top_k = st.sidebar.slider(
    "Retrieve Top-K Chunks",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Number of most relevant text chunks to feed to the language model."
)

custom_system_prompt = st.sidebar.text_area(
    "Custom System Prompt",
    value="You are a helpful assistant. Use the provided context to answer the user's question. If the answer cannot be found in the context, say that you don't know based on the available documents. Keep your answer professional and grounded.",
    height=120,
    help="Control the guidelines and behavior of the generator LLM."
)

# Index reset helper
if st.sidebar.button("🗑️ Reset Database & Chat"):
    st.session_state.documents = {}
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.session_state.telemetry = {
        "ingest_time": 0.0,
        "chunk_count": 0,
        "embed_time": 0.0,
        "search_time": 0.0,
        "generate_time": 0.0
    }
    st.toast("Database index and conversations reset successfully!")
    st.rerun()


# ==========================================
# 4. Main Page View
# ==========================================
st.markdown("<h1 style='text-align: center; color: #6366f1; margin-bottom: 5px;'>🧠 RAG Document QA Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 16px; margin-bottom: 25px;'>Upload private notes, resumes, books, or research papers and query them with semantic accuracy.</p>", unsafe_allow_html=True)

# Tabs
tab_chat, tab_inspector = st.tabs(["💬 Chat Q&A Interface", "🔍 Document & Chunk Inspector"])

# -----------------
# TAB 1: CHAT & RAG
# -----------------
with tab_chat:
    # Top Action Panel: Ingestion
    with st.expander("📂 Ingest Custom Documents (PDF / TXT)", expanded=True):
        col_up, col_btn = st.columns([4, 1])
        with col_up:
            uploaded_files = st.file_uploader(
                "Select PDF or TXT files to add to the RAG database",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
        with col_btn:
            process_btn = st.button("⚡ Build Index", use_container_width=True)

        # Handle processing
        if process_btn and uploaded_files:
            start_ingest = time.time()
            progress_bar = st.progress(0, text="Extracting texts...")
            
            # Temporary directory setup
            os.makedirs("temp_uploads", exist_ok=True)
            
            all_new_chunks = []
            all_new_metadata = []
            
            total_files = len(uploaded_files)
            for i, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                
                # Check if already ingested to avoid re-parsing same parameters
                # (Re-parsing is done if chunk parameters change, so we reconstruct)
                temp_path = os.path.join("temp_uploads", file_name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(int((i + 0.3) / total_files * 100), text=f"Parsing: {file_name}...")
                
                try:
                    # Extraction
                    raw_text = extract_text(temp_path)
                    
                    # Chunking
                    chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
                    
                    # Save document record
                    st.session_state.documents[file_name] = {
                        "text": raw_text,
                        "chunks": chunks,
                        "char_count": len(raw_text)
                    }
                    
                    # Build chunks registry for Vector Database
                    for idx, chunk in enumerate(chunks):
                        all_new_chunks.append(chunk)
                        all_new_metadata.append({
                            "source": file_name,
                            "chunk_index": idx,
                            "text": chunk
                        })
                except Exception as e:
                    st.error(f"Error parsing {file_name}: {str(e)}")
                
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            ingest_elapsed = time.time() - start_ingest
            st.session_state.telemetry["ingest_time"] = ingest_elapsed
            st.session_state.telemetry["chunk_count"] = len(all_new_chunks)
            
            # Generate Embeddings & Build Vector Database
            progress_bar.progress(80, text="Creating embeddings and building vector index...")
            
            start_embed = time.time()
            try:
                # Initialize embedding service
                emb_service = EmbeddingService(provider=embedding_provider, api_key=api_key)
                
                # Generate embeddings for all chunks
                vectors = emb_service.embed_texts(all_new_chunks, is_query=False)
                
                # Instantiate Vector Store (with FAISS vs NumPy fallback based on availability)
                # We can prefer FAISS if available, but auto fallback to False if missing
                st.session_state.vector_store = RAGVectorStore(
                    dimension=embedding_dim,
                    use_faiss=HAS_FAISS
                )
                st.session_state.vector_store.add_vectors(vectors, all_new_metadata)
                
                embed_elapsed = time.time() - start_embed
                st.session_state.telemetry["embed_time"] = embed_elapsed
                
                progress_bar.progress(100, text="Index built successfully!")
                time.sleep(1)
                progress_bar.empty()
                st.success(f"Indexed {len(uploaded_files)} document(s) into {len(all_new_chunks)} text chunks using {st.session_state.vector_store.use_faiss and 'FAISS' or 'NumPy (Fallback)'} Vector Store!")
            except Exception as e:
                progress_bar.empty()
                st.error(f"Failed to generate embeddings: {str(e)}")

    # Telemetry Panel
    tel = st.session_state.telemetry
    st.markdown("### 📊 Performance Diagnostic Panel")
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    with col_t1:
        st.markdown(f"<div class='telemetry-card'><div class='telemetry-value'>{tel['ingest_time']:.3f}s</div><div class='telemetry-label'>Text Extraction</div></div>", unsafe_allow_html=True)
    with col_t2:
        st.markdown(f"<div class='telemetry-card'><div class='telemetry-value'>{tel['chunk_count']}</div><div class='telemetry-label'>Total Chunks</div></div>", unsafe_allow_html=True)
    with col_t3:
        st.markdown(f"<div class='telemetry-card'><div class='telemetry-value'>{tel['embed_time']:.3f}s</div><div class='telemetry-label'>Embedding Time</div></div>", unsafe_allow_html=True)
    with col_t4:
        st.markdown(f"<div class='telemetry-card'><div class='telemetry-value'>{tel['search_time']:.3f}s</div><div class='telemetry-label'>Retrieval Time</div></div>", unsafe_allow_html=True)
    with col_t5:
        st.markdown(f"<div class='telemetry-card'><div class='telemetry-value'>{tel['generate_time']:.3f}s</div><div class='telemetry-label'>Generation Time</div></div>", unsafe_allow_html=True)

    # Chat Area
    st.markdown("---")
    
    # Informative notice if no documents are loaded
    if not st.session_state.documents:
        st.info("💡 **Welcome!** Please upload PDF/TXT documents and click **Build Index** to start querying. Alternatively, configure your API keys in the sidebar.")
    
    # Display Chat History
    for chat in st.session_state.chat_history:
        # User message
        st.markdown(f"<div class='chat-bubble chat-user'><b>You:</b><br>{chat['query']}</div>", unsafe_allow_html=True)
        # Assistant message
        st.markdown(f"<div class='chat-bubble chat-assistant'><b>AI Assistant:</b><br>{chat['answer']}</div>", unsafe_allow_html=True)
        
        # Display sources as expander cards if they exist
        if chat.get("sources"):
            with st.expander(f"📚 Grounded Context (Similarity Sources: {len(chat['sources'])})"):
                for idx, src in enumerate(chat["sources"]):
                    st.markdown(f"""
                    <div class='source-card'>
                        <div class='source-header'>
                            <span>📄 Source: <b>{src['source']}</b> (Chunk #{src['chunk_index']})</span>
                            <span>🎯 Score: <b>{src['score']:.4f}</b></span>
                        </div>
                        <div style='font-size: 13.5px; line-height: 1.4; color: #cbd5e1;'>
                            <i>"{src['text']}"</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Chat Input Box
    query_input = st.chat_input("Ask a question about the loaded documents...")
    if query_input:
        if not st.session_state.documents or not st.session_state.vector_store:
            st.warning("⚠️ Please upload and index documents first before asking questions!")
        else:
            # Process query
            with st.spinner("Processing query..."):
                # 1. Embed query
                start_search = time.time()
                try:
                    emb_service = EmbeddingService(provider=embedding_provider, api_key=api_key)
                    query_vector = emb_service.embed_texts([query_input], is_query=True)[0]
                    
                    # 2. Retrieve relevant chunks
                    retrieved_chunks, scores = st.session_state.vector_store.search(query_vector, k=top_k)
                    search_elapsed = time.time() - start_search
                    st.session_state.telemetry["search_time"] = search_elapsed
                except Exception as e:
                    st.error(f"Error during retrieval embedding: {str(e)}")
                    st.stop()

                # Extract chunk texts for LLM prompt
                context_texts = [chunk["text"] for chunk in retrieved_chunks]
                
                # 3. Generate response
                start_gen = time.time()
                try:
                    answer = generate_answer(
                        query=query_input,
                        contexts=context_texts,
                        provider=llm_provider,
                        api_key=api_key,
                        model_name=llm_model,
                        system_prompt=custom_system_prompt
                    )
                    gen_elapsed = time.time() - start_gen
                    st.session_state.telemetry["generate_time"] = gen_elapsed
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
                    st.session_state.telemetry["generate_time"] = 0.0
                
                # Pack sources metadata
                sources = []
                for chunk, score in zip(retrieved_chunks, scores):
                    sources.append({
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "score": score
                    })
                
                # Save chat record
                st.session_state.chat_history.append({
                    "query": query_input,
                    "answer": answer,
                    "sources": sources
                })
                
                st.rerun()

# ----------------------------
# TAB 2: DOCUMENT INSPECTOR
# ----------------------------
with tab_inspector:
    st.markdown("### 🔍 Ingested Document Overview")
    if not st.session_state.documents:
        st.info("No documents have been loaded yet. Upload and build index to inspect chunks.")
    else:
        # Document Stats Table
        docs_summary = []
        for name, data in st.session_state.documents.items():
            docs_summary.append({
                "Document Name": name,
                "Characters": data["char_count"],
                "Total Chunks": len(data["chunks"])
            })
        st.dataframe(docs_summary, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔎 Text Chunk Inspector")
        st.write("Understand how the text chunks look after splitting based on the current chunk size/overlap parameters.")

        # Select file to inspect
        selected_file = st.selectbox(
            "Select document to inspect:",
            list(st.session_state.documents.keys())
        )

        if selected_file:
            doc_data = st.session_state.documents[selected_file]
            chunks = doc_data["chunks"]
            total_c = len(chunks)
            
            if total_c == 0:
                st.warning("This document has no chunks.")
            else:
                chunk_idx = st.slider(
                    "Select Chunk Index to View:",
                    min_value=0,
                    max_value=total_c - 1,
                    value=0,
                    step=1,
                    help="Slide to navigate through successive chunks of the document."
                )

                col_chunk_meta, col_chunk_text = st.columns([1, 3])
                with col_chunk_meta:
                    st.markdown(f"""
                    <div style='background-color: #111827; border: 1px solid #1f2937; padding: 16px; border-radius: 8px;'>
                        <p style='margin: 0; color: #94a3b8; font-size: 12px;'>DOCUMENT</p>
                        <p style='margin: 0 0 10px 0; font-weight: bold; color: #f3f4f6;'>{selected_file}</p>
                        <p style='margin: 0; color: #94a3b8; font-size: 12px;'>CHUNK INDEX</p>
                        <p style='margin: 0 0 10px 0; font-weight: bold; color: #38bdf8;'>{chunk_idx} of {total_c - 1}</p>
                        <p style='margin: 0; color: #94a3b8; font-size: 12px;'>CHUNK SIZE</p>
                        <p style='margin: 0; font-weight: bold; color: #34d399;'>{len(chunks[chunk_idx])} characters</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_chunk_text:
                    st.text_area(
                        "Chunk Content:",
                        value=chunks[chunk_idx],
                        height=200,
                        disabled=True
                    )
