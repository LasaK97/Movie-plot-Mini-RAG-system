import faiss
import numpy as np
from typing import List, Dict

from src.config import Config

class VectorStore:
    """FAISS based vector store."""

    def __init__(self, config: Config, dimension: int):
        self.config = config
        self.dimension = dimension
        self.index = None
        self.chunks = []

        if config.system.verbose:
            print(f"Initializing FAISS index (dimension = {self.dimension})...")

        self.create_index()

    def create_index(self):
        """create faiss index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.config.system.verbose:
            print(f"FAISS index created: {self.config.vector_store.index_type}")

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """add embeddings to index"""
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Number of embeddings ({len(embeddings)}) must match "
                f"number of chunks ({len(chunks)})"
            )

        if self.config.system.verbose:
            print(f"Adding {len(embeddings)} embeddings to index...")
        # add to faiss index
        self.index.add(embeddings.astype('float32'))

        #store chunk metadata
        self.chunks.extend(chunks)

        if self.config.system.verbose:
            print(f"Index now contains {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, k: int = None) -> List[Dict]:
        """search for similar vectors"""

        if self.index.ntotal ==0:
            raise ValueError("Index is empty. Add embeddings first.")

        if k is None:
            k = self.config.retrieval.top_k

        #ensure k  less than index size
        k = min(k, self.index.ntotal)

        #ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        #search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        #similarities
        similarities = 1 - (distances[0] / 2)

        #results
        results = []
        for idx, similarity, distance in zip(indices[0], similarities, distances[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(similarity)
                result['distance'] = float(distance)
                result['index'] = int(idx)

                if similarity >= self.config.retrieval.min_similarity:
                    results.append(result)

        return results

    def get_stats(self) -> Dict:
        """get statistics of vector store"""
        unique_movies = len(set(chunk['movie_id'] for chunk in self.chunks))

        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'total_chunks': len(self.chunks),
            'unique_movies': unique_movies,
            'index_type': self.config.vector_store.index_type
        }