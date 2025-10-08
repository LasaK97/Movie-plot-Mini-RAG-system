import os
from pathlib import Path
from typing import Optional, List
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# load env
load_dotenv(verbose=True)

PROJECT_ROOT = Path(__file__).parent.parent

class DataFilters(BaseModel):
    """Filters dataset"""
    min_plot_length: int = Field(100, ge=1)
    year_from: int = Field(2000, ge=1900)
    year_to: Optional[int] = Field(None, ge=2010)

    @validator("year_to")
    def year_to_after_year_from(cls, v, values):
        """validates the year to is after year from"""
        if v is not None and 'year_from' in values and v < values['year_from']:
            raise ValueError("year_to must be greater than or equal to year_from")
        return v

class DataConfig(BaseModel):
    """Data loading and filtering configs"""
    dataset_path: str
    sample_size: int = Field(300, ge=1)
    filters: DataFilters
    random_seed: int = 42

class ChunkingConfig(BaseModel):
    """Chunking configs"""
    size: int = Field(300, ge=50, le=1000)
    overlap: int = Field(50, ge=0)
    min_chunk_size: int = Field(50, ge=10)

    @validator("overlap")
    def overlap_less_than_size(cls, v, values):
        """validates the overlap is less than the chunk size"""
        if 'size' in values and v >= values['size']:
            raise ValueError("overlap must be less than the size")
        return v

class TextProcessingConfig(BaseModel):
    """Text processor configs"""
    chunking: ChunkingConfig

class EmbeddingsConfig(BaseModel):
    """Embedding generation configs"""
    model: str = "all-MiniLM-L6-v2"
    dimension: int = Field(384, ge=1)
    batch_size: int = Field(32, ge=1)
    normalize: bool = True
    show_progress: bool = True

class VectorStoreConfig(BaseModel):
    """Vector store configs"""
    index_type: str = "IndexFlatL2"
    metric: str = Field("l2", pattern="^(l2|cosine)$")

class LLMParameters(BaseModel):
    """LLM generation parameters"""
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(500, ge=1)
    timeout: int = Field(30, ge=1)

class LLMRetry(BaseModel):
    """LLM retry configs"""
    max_attempts: int = Field(3, ge=1)
    backoff_factor: int = Field(2, ge=1)

class LLMConfig(BaseModel):
    """LLM generation configs"""
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_url: str
    parameters: LLMParameters
    retry: LLMRetry

class RetrievalConfig(BaseModel):
    """Retrieval configs"""
    top_k: int = Field(3, ge=1)
    max_context_length: int = Field(400, ge=50)
    min_similarity: float = Field(0.0, ge=0.0, le=1.0)

class SystemConfig(BaseModel):
    """System configs"""
    verbose: bool = True
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

class Config(BaseModel):
    """main config object"""
    data: DataConfig
    text_processing: TextProcessingConfig
    embedding: EmbeddingsConfig
    vector_store: VectorStoreConfig
    llm: LLMConfig
    retrieval: RetrievalConfig
    system: SystemConfig

    def get_dataset_path(self) -> Path:
        """get the dataset abs path"""
        return PROJECT_ROOT / self.data.dataset_path

    def get_api_key(self) -> str:
        """get api key from env """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return api_key


def load_config(config_path: Optional[Path]= None) -> Config:
    """load config from path"""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    try:
        config = Config(**config_dict)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")



def print_config(config: Config) -> None:
    """summary of the configs"""

    header = "CONFIGURATION SUMMARY"
    header_line = "=" * 70

    #Config summary
    config_summary = f"""
    {header_line}
    {header.center(70)}
    {header_line}
    
    [Data]
    Dataset             : {config.data.dataset_path}
    Sample size         : {config.data.sample_size}
    Min plot length     : {config.data.filters.min_plot_length} words
    Year Range          : {config.data.filters.year_from}-{config.data.filters.year_to or 'present'}
    
    [Text Processing]
    chunk_size          : {config.text_processing.chunking.size} words 
    chunk_overlap       : {config.text_processing.chunking.overlap} words 
    
    [Embeddings]
    Model               : {config.embedding.model} 
    Dimension           : {config.embedding.dimension} 
    Batch size          : {config.embedding.batch_size}  
    
    [Vector Store]
    Index type          : {config.vector_store.index_type}
    Metric              : {config.vector_store.metric} 
    
    [LLM]
    Provider            : {config.llm.provider}
    Model               : {config.llm.model} 
    Temperature         : {config.llm.parameters.temperature}
    Max tokens          : {config.llm.parameters.max_tokens}
    
    [Retrieval]
    Top-K               : {config.retrieval.top_k}
    Max context length  : {config.retrieval.max_context_length} chars
    
    {header_line}
    """

    print(config_summary)


config: Optional[Config] = None

try:
    config = load_config()
except Exception as e:
    print(f"Warning: Could not load config: {e}")


