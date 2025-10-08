import re
from typing import List, Dict
import pandas as pd

from src.config import Config

def preprocess_text(text: str) -> str:
    """clean and preprocess text"""
    text = ' '.join(text.split())

    #remove special chars
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int, min_chunk_size:int) -> List[str]:
    """splits text into overlapping chunks based on word count"""

    words = text.split()

    #check the text is shorter than the chunk size
    if len(words) <= chunk_size:
        return [text] if len(words) >= min_chunk_size else []

    chunks = []
    start = 0

    while start < len(words):
        #get chunk
        end = start + chunk_size
        chunk_words = words[start:end]

        #skip too small chunks
        if len(chunk_words) >= min_chunk_size:
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)

        #move start with overlap
        start += chunk_size - overlap

        #end
        if end >= len(words):
            break

    return chunks


def create_chunks_with_metadata(df: pd.DataFrame, config: Config) -> List[Dict]:
    """creates chunks with metadata"""
    chunking = config.text_processing.chunking

    if config.system.verbose:
        print(f"Creating text chunks (size= {chunking.size}, overlap= {chunking.overlap})")

    all_chunks = []

    for idx, row in df.iterrows():
        title = row["Title"]
        plot = preprocess_text(row["Plot"])

        chunks = chunk_text(
            plot,
            chunk_size=chunking.size,
            overlap=chunking.overlap,
            min_chunk_size=chunking.min_chunk_size,
        )

        #add metadata
        for chunk_idx, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'movie_title': title,
                'movie_id': int(idx),
                'chunk_id': chunk_idx,
                'total_chunks': len(chunks)
            }

            if 'Release Year' in row and pd.notna(row['Release Year']):
                chunk_data['release_year'] = int(row['Release Year'])

            if 'Genre' in row and pd.notna(row['Genre']):
                chunk_data['genre'] = row['Genre']

            if 'Director' in row and pd.notna(row['Director']):
                chunk_data['director'] = row['Director']

            all_chunks.append(chunk_data)

    if config.system.verbose:
        print(f"Created {len(all_chunks)} chunks from {len(df)} movies")
        avg_chunks = len(all_chunks) / len(df) if len(df) > 0 else 0
        print(f"Average chunks per movie: {avg_chunks:.2f}")

    return all_chunks


def get_chunk_stats(chunks: List[Dict]) -> Dict:
    """get statistics about the chunks"""
    if not chunks:
        return {
            'total_chunks': 0,
            'avg_chunk_length': 0,
            'min_chunk_length': 0,
            'max_chunk_length': 0,
            'unique_movies': 0
        }

    chunk_lengths = [len(chunk['text'].split()) for chunk in chunks]

    return {
        'total_chunks': len(chunks),
        'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
        'min_chunk_length': min(chunk_lengths),
        'max_chunk_length': max(chunk_lengths),
        'unique_movies': len(set(chunk['movie_id'] for chunk in chunks))
    }