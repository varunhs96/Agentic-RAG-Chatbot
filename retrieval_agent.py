import faiss
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

class RetrievalAgent:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.chunk_store = []
        self.is_built = False
    
    def build_index(self, documents: Dict[str, List[str]]):
        """Build FAISS index from documents"""
        all_chunks = []
        
        # Collect all valid chunks
        for doc_name, chunks in documents.items():
            for chunk in chunks:
                if chunk and chunk.strip():  # Check for both None and empty strings
                    all_chunks.append((doc_name, chunk.strip()))
        
        if not all_chunks:
            raise ValueError("No valid chunks found to embed.")
        
        print(f"Processing {len(all_chunks)} chunks...")
        
        # Store chunks
        self.chunk_store = all_chunks
        texts = [chunk for _, chunk in all_chunks]
        
        # Generate embeddings
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Clear previous index and add new embeddings
            self.index.reset()
            self.index.add(embeddings)
            self.is_built = True
            
            print(f"Index built successfully with {self.index.ntotal} vectors")
            
        except Exception as e:
            raise RuntimeError(f"Failed to build index: {str(e)}")
    
    def query(self, question: str, top_k: int = 5, similarity_threshold: float = None) -> List[Dict]:
        """Query the index for similar chunks"""
        if not self.is_built or self.index.ntotal == 0:
            raise RuntimeError("Index not built or empty. Call build_index() first.")
        
        if not question or not question.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            # Generate query embedding
            query_vector = self.embedding_model.encode([question.strip()])
            query_vector = np.array(query_vector, dtype=np.float32)
            
            # Search
            distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.chunk_store):  # Valid index check
                    distance = float(distances[0][i])
                    similarity = 1 / (1 + distance)  # Convert L2 distance to similarity
                    
                    # Apply similarity threshold if provided
                    if similarity_threshold and similarity < similarity_threshold:
                        continue
                    
                    doc_name, chunk_text = self.chunk_store[idx]
                    results.append({
                        "doc": doc_name,
                        "chunk": chunk_text,
                        "similarity": similarity,
                        "distance": distance
                    })
            
            print(f"Found {len(results)} results for query: '{question[:50]}...'")
            return results
            
        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the current index"""
        return {
            "total_chunks": len(self.chunk_store),
            "index_size": self.index.ntotal,
            "vector_dimension": self.vector_dim,
            "is_built": self.is_built
        }