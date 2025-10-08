import time
from typing import Dict, List, Optional

from src.config import Config
from src.data_loader import load_dataset
from src.text_processor import create_chunks_with_metadata
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_client import LLMClient
from src.utils import format_context

class RAGPipeline:
    """RAG system pipeline"""
    def __init__(self, config: Config):
        self.config = config

        if config.system.verbose:
            print("\n" + "=" * 70)
            print("INITIALIZING RAG SYSTEM")
            print("=" * 70)

        start_time = time.time()

        #init components
        self.embedding_generator = None
        self.vector_store = None
        self.llm_client = None
        self.chunks = []

        #setup system
        self.setup()

        init_time = time.time() - start_time
        if config.system.verbose:
            print(f"\n Successfully initialized RAG System in {init_time:.2f} seconds.")
            print("=" * 70 + "\n")

    def setup(self):
        """setup components"""

        # 1. load and process data
        if self.config.system.verbose:
            print("\n[1/5] Loading dataset...")
        df = load_dataset(self.config)

        # 2. create chunks
        if self.config.system.verbose:
            print("\n[2/5] Creating text chunks...")
        self.chunks = create_chunks_with_metadata(df, self.config)

        #3. initialize the embedding generator
        if self.config.system.verbose:
            print("\n[3/5] Loading embedding model...")
        self.embedding_generator = EmbeddingGenerator(self.config)

        #4. generate embeddings and build index
        if self.config.system.verbose:
            print("\n[4/5] Generating embeddings and building index...")
        embeddings = self.embedding_generator.embed_chunks(self.chunks)
        self.vector_store = VectorStore(
            self.config,
            dimension=self.embedding_generator.dimension,
        )
        self.vector_store.add_embeddings(embeddings, self.chunks)

        #5. initialize the LLM client
        if self.config.system.verbose:
            print("\n[5/5] Initializing LLM client...")
        self.llm_client = LLMClient(self.config)

    def query(self, query: str, verbose: Optional[bool] = None) -> Dict:
        """query the RAG system"""
        if verbose is None:
            verbose = self.config.system.verbose

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"QUERY: {query}")
            print('=' * 70)

        start_time = time.time()

        try:
            #1. retrieve relevant contexts
            if verbose:
                print("\n[1/3] Retrieving relevant contexts...")
            retrieved_chunks = self.retrieve_contexts(query)

            if verbose:
                print(f"Retrieved {len(retrieved_chunks)} contexts")
                for i, chunk in enumerate(retrieved_chunks, 1):
                    print(f"  {i}. {chunk['movie_title']} | (score: {chunk['similarity_score']:.3f})")

            # 2. format contexts
            contexts = format_context(
                retrieved_chunks,
                max_chars=self.config.retrieval.max_context_length
            )

            #3. generating answer
            if verbose:
                print("\n[2/3] Generating answer with LLM...")

            llm_result = self.llm_client.generate_answer(
                contexts,
                query
            )

            if not llm_result['success']:
                raise Exception(llm_result.get('error', 'LLM generation failed'))

            #4. generate reasoning
            if verbose:
                print("\n[3/3] Generating reasoning...")
            reasoning = self.llm_client.generate_reasoning(
                contexts,
                query,
                llm_result['answer']
            )

            #5. construct response
            query_time = time.time() - start_time

            response = {
                'query': query,
                'answer': llm_result['answer'],
                'contexts': contexts,
                'reasoning': reasoning,
                'metadata': {
                    'num_contexts': len(contexts),
                    'top_similarity': (retrieved_chunks[0]['similarity_score'] if retrieved_chunks else 0),
                    'query_time': query_time,
                    'model': llm_result.get('model', 'unknown'),
                    'usage': llm_result.get('usage', {})
                }
            }

            if verbose:
                print(f"\nQuery completed in {query_time:.2f} seconds")

            return response

        except Exception as e:
            query_time = time.time() - start_time

            if verbose:
                print(f"\n Error processing query: {e}")

            return {
                'query': query,
                'answer': f"Error: {str(e)}",
                'contexts': [],
                'reasoning': "Query failed due to an error",
                'metadata': {
                    'error': str(e),
                    'query_time': query_time
                }
            }

    def retrieve_contexts(self, query: str) -> List[Dict]:
        """retrieve contexts for query"""
        #generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)

        #search vector store
        results = self.vector_store.search(
            query_embedding,
            k = self.config.retrieval.top_k
        )

        return results

    def get_stats(self) -> Dict:
        """get statistics of the rag pipeline"""
        return {
            'total_chunks': len(self.chunks),
            'unique_movies': len(set(chunk['movie_id'] for chunk in self.chunks)),
            'embedding_dimension': self.embedding_generator.dimension,
            'embedding_model': self.config.embedding.model,
            'vector_store_size': self.vector_store.index.ntotal,
            'llm_model': self.config.llm.model,
            'llm_provider': self.config.llm.provider
        }


