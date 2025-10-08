import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm

from src.config import Config

class EmbeddingGenerator:
    """generates embeddings"""

    def __init__(self,config: Config):
        self.config = config
        self.model = None
        self.dimension = config.embedding.dimension

        if config.system.verbose:
            print(f"Loading embedding model: {config.embedding.model}...")

        self.model = SentenceTransformer(config.embedding.model)

        #verify
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != self.dimension:
            print(f"Warning: Config dimension ({self.dimension}) doesn't match "
                  f"model dimension ({actual_dim}). Using model dimension.")
            self.dimension = actual_dim

        if config.system.verbose:
            print(f"Model loaded. Embedding dimension: {self.dimension}")

    def generate_embeddings(self,texts: List[str]) -> np.ndarray:
        """generate embeddings for list of text"""
        if self.config.system.verbose:
            print(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=self.config.embedding.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.embedding.normalize
        )

        if self.config.system.verbose:
            print(f"Embedding generated: shape: {embeddings.shape}")

        return embeddings

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """generate embeddings for text chunks"""
        texts = [chunk['text'] for chunk in chunks]
        return self.generate_embeddings(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """generate embeddings for query"""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.config.embedding.normalize
        )
        return embedding[0]


