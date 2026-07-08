import os
import time
import json
import pickle
import numpy as np
from pypdf import PdfReader

# Optional FAISS import with fallback flag
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Optional SentenceTransformers import
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# ==========================================
# 1. Document Extraction & Processing
# ==========================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Error reading PDF {pdf_path}: {str(e)}")
    return text

def extract_text_from_txt(txt_path: str) -> str:
    """Extracts text from a plain text file."""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading TXT {txt_path}: {str(e)}")

def extract_text(file_path: str) -> str:
    """Detects file type and extracts raw text."""
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF and TXT are supported.")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Splits a document text into smaller chunks with overlap, avoiding breaking words."""
    if chunk_size <= 0:
        chunk_size = 500
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break

        # Look back 20% of chunk_size to find a clean boundary
        lookback_limit = max(start, end - chunk_size // 5)
        boundary_range = text[lookback_limit:end]
        found_boundary = -1
        
        # Priority boundaries: double newline, single newline, punctuation spacing, space
        for marker in ['\n\n', '\n', '. ', '? ', '! ', ' ']:
            idx = boundary_range.rfind(marker)
            if idx != -1:
                found_boundary = lookback_limit + idx + len(marker)
                break

        if found_boundary != -1:
            end = found_boundary

        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
        if start < 0:
            start = 0

    return [c for c in chunks if c.strip()]


# ==========================================
# 2. Embedding Service Wrapper
# ==========================================

class EmbeddingService:
    def __init__(self, provider="local", api_key=None):
        self.provider = provider
        self.api_key = api_key
        self.local_model = None

        if provider == "local":
            if HAS_SENTENCE_TRANSFORMERS:
                # Lazy loading of sentence transformer model
                self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            else:
                print("Warning: sentence-transformers not installed. Fallback to basic term hashing.")

    def embed_texts(self, texts: list[str], is_query=False) -> list[list[float]]:
        if not texts:
            return []

        if self.provider == "google":
            if not self.api_key:
                raise ValueError("Google API key is required for Gemini embeddings.")
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            task_type = "retrieval_query" if is_query else "retrieval_document"
            response = genai.embed_content(
                model="models/text-embedding-004",
                contents=texts,
                task_type=task_type
            )
            # Response format check (it can be single embedding or a list depending on type)
            embeddings = response.get("embedding", [])
            # If embedding is a single list (e.g. for single item), wrap it
            if embeddings and not isinstance(embeddings[0], list):
                return [embeddings]
            return embeddings

        elif self.provider == "cohere":
            if not self.api_key:
                raise ValueError("Cohere API key is required for Cohere embeddings.")
            import cohere
            co = cohere.Client(api_key=self.api_key)
            input_type = "search_query" if is_query else "search_document"
            response = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type=input_type
            )
            # Cohere 5.x returns response.embeddings
            return [list(emb) for emb in response.embeddings]

        else:  # Local mode
            if self.local_model:
                embeddings = self.local_model.encode(texts)
                return [emb.tolist() for emb in embeddings]
            else:
                # Basic Term Hashing/frequency fallback if dependencies are completely missing (offline fallback)
                return [self._basic_hash_vector(text) for text in texts]

    def _basic_hash_vector(self, text: str, dim: int = 384) -> list[float]:
        """A simple, dependency-free deterministic hashing vector generator for offline fallback."""
        vec = np.zeros(dim)
        words = text.lower().split()
        for w in words:
            # Simple hash mapping to dimension
            h = hash(w) % dim
            vec[h] += 1.0
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# ==========================================
# 3. Dual-Layer Vector Databases (FAISS & NumPy)
# ==========================================

class NumpyVectorStore:
    """A pure Python/NumPy implementation of a Vector Store for zero binary dependencies."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []  # List of list[float]
        self.metadata = [] # List of dict

    def add(self, vectors: list[list[float]], metadata: list[dict]):
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: list[float], k: int = 3) -> tuple[list[dict], list[float]]:
        if not self.vectors:
            return [], []
        
        vectors_arr = np.array(self.vectors)  # (N, D)
        query_arr = np.array(query_vector)    # (D,)

        # Compute cosine similarity
        norms = np.linalg.norm(vectors_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vectors = vectors_arr / norms

        q_norm = np.linalg.norm(query_arr)
        if q_norm == 0:
            q_norm = 1.0
        norm_query = query_arr / q_norm

        scores = np.dot(norm_vectors, norm_query) # (N,)
        top_k_indices = np.argsort(scores)[::-1][:k]

        retrieved_metadata = [self.metadata[idx] for idx in top_k_indices]
        retrieved_scores = [float(scores[idx]) for idx in top_k_indices]
        return retrieved_metadata, retrieved_scores


class RAGVectorStore:
    """Wraps FAISS index with NumpyVectorStore fallback."""
    def __init__(self, dimension: int, use_faiss: bool = True):
        self.dimension = dimension
        self.use_faiss = use_faiss and HAS_FAISS
        self.metadata = []

        if self.use_faiss:
            # We use L2 distance for FAISS IndexFlatL2
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = NumpyVectorStore(dimension)

    def add_vectors(self, vectors: list[list[float]], metadata: list[dict]):
        if not vectors:
            return

        self.metadata.extend(metadata)
        vectors_arr = np.array(vectors, dtype=np.float32)

        if self.use_faiss:
            # FAISS expects float32 numpy array
            self.index.add(vectors_arr)
        else:
            self.index.add(vectors, metadata)

    def search(self, query_vector: list[float], k: int = 3) -> tuple[list[dict], list[float]]:
        if not self.metadata:
            return [], []

        query_arr = np.array([query_vector], dtype=np.float32)

        if self.use_faiss:
            # D = distances, I = indices
            k_adjusted = min(k, len(self.metadata))
            D, I = self.index.search(query_arr, k_adjusted)
            
            results = []
            scores = []
            for dist, idx in zip(D[0], I[0]):
                if idx != -1 and idx < len(self.metadata):
                    results.append(self.metadata[idx])
                    # Convert L2 distance to a standard similarity score: 1 / (1 + L2)
                    sim_score = 1.0 / (1.0 + float(dist))
                    scores.append(sim_score)
            return results, scores
        else:
            return self.index.search(query_vector, k)

    def save(self, filepath: str):
        """Saves vector store to disk."""
        data = {
            "dimension": self.dimension,
            "use_faiss": self.use_faiss,
            "metadata": self.metadata
        }
        
        if self.use_faiss:
            # Serialize FAISS index to bytes
            faiss_bytes = faiss.serialize_index(self.index)
            data["faiss_bytes"] = faiss_bytes
        else:
            data["numpy_vectors"] = self.index.vectors

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "RAGVectorStore":
        """Loads vector store from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        dimension = data["dimension"]
        use_faiss = data["use_faiss"]
        
        # Fallback check
        if use_faiss and not HAS_FAISS:
            print("Warning: Saved store requires FAISS, but it is not installed. Loading in Numpy fallback mode.")
            use_faiss = False

        store = cls(dimension, use_faiss=use_faiss)
        store.metadata = data["metadata"]

        if use_faiss:
            store.index = faiss.deserialize_index(data["faiss_bytes"])
        else:
            numpy_store = NumpyVectorStore(dimension)
            numpy_store.vectors = data.get("numpy_vectors", [])
            numpy_store.metadata = store.metadata
            store.index = numpy_store

        return store


# ==========================================
# 4. LLM Generation Connectors
# ==========================================

def generate_answer(
    query: str,
    contexts: list[str],
    provider: str = "mock",
    api_key: str = None,
    model_name: str = None,
    system_prompt: str = None
) -> str:
    """Generates an answer from context grounding using selected LLM."""
    if not system_prompt:
        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question. "
            "If the answer cannot be found in the context, say that you don't know based on the "
            "available documents. Keep your answer professional and grounded."
        )

    # Format the retrieved chunks into a clear prompt block
    context_str = "\n\n".join([f"--- SOURCE CHUNK {i+1} ---\n{c}" for i, c in enumerate(contexts)])
    
    prompt = f"""System Instruction:
{system_prompt}

Context Information:
{context_str}

User Question: {query}

Grounded Answer:"""

    if provider == "google":
        if not api_key:
            return "Error: Google API key is missing. Please enter it in the sidebar."
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model_to_use = model_name if model_name else "gemini-1.5-flash"
        model = genai.GenerativeModel(
            model_name=model_to_use,
            system_instruction=system_prompt
        )
        
        # Format the user prompt
        user_prompt = f"Context Information:\n{context_str}\n\nUser Question: {query}"
        response = model.generate_content(user_prompt)
        return response.text

    elif provider == "cohere":
        if not api_key:
            return "Error: Cohere API key is missing. Please enter it in the sidebar."
        import cohere
        co = cohere.Client(api_key=api_key)
        
        model_to_use = model_name if model_name else "command-r-plus"
        
        # Cohere supports structured documents in their chat API
        documents = [{"title": f"Source {i+1}", "snippet": c} for i, c in enumerate(contexts)]
        
        response = co.chat(
            message=query,
            documents=documents,
            model=model_to_use,
            preamble=system_prompt
        )
        return response.text

    else:  # Mock offline response
        # Detailed, realistic breakdown for demonstration when offline
        time.sleep(1.0) # Simulate network lag
        sources_list = "\n".join([f"- Chunk {i+1} (approx {len(c)} chars)" for i, c in enumerate(contexts)])
        
        mock_response = f"""**[Demo Mode - Offline]**

You queried: *"{query}"*

The system retrieved the following relevant document resources:
{sources_list}

**Standard System Behavior Description:**
When connected to an active API key (Gemini/Cohere), this query and the retrieved snippets are compiled into a prompt template. The LLM then answers the question using only the facts presented.

**Snippet Previews:**
"""
        for i, c in enumerate(contexts[:2]):
            preview = c[:120].replace('\n', ' ') + "..." if len(c) > 120 else c
            mock_response += f"\n* **Chunk {i+1}:** \"{preview}\""
            
        return mock_response
