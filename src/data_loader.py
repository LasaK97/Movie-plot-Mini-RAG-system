import pandas as pd
from typing import Optional
from pathlib import Path

from src.config import Config

def load_dataset(config: Config) -> pd.DataFrame:
    """load and filter the dataset"""
    dataset_path = config.get_dataset_path()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if config.system.verbose:
        print(f"""Loaded dataset from {dataset_path} 
        Original dataset size: {len(df)} movies.
                """)

    #validate and filter
    df = validate_data(df, config)
    df = filter_data(df, config)

    #sample
    sample_size = config.data.sample_size
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=config.data.random_seed)
        if config.system.verbose:
            print(f"Sampled {sample_size} movies")

    #reset index
    df =df.reset_index(drop=True)

    if config.system.verbose:
        print(f"Final dataset size: {len(df)} movies")

    return df

def validate_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """validate the data frame"""
    if config.system.verbose:
        print(f"Validating data ...")

    required_cols = ['Title', 'Plot']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    #removes rows with missing title ot plot
    initial_len = len(df)
    df =df.dropna(subset=required_cols)

    #remove empty strings
    df = df[df['Title'].str.strip() != '']
    df = df[df['Plot'].str.strip() != '']

    removed = initial_len - len(df)
    if config.system.verbose:
        print(f"Removed {removed} rows with missing data.")

    return df

def filter_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """filter the data frame"""
    if config.system.verbose:
        print(f"Filtering data ...")

    filters = config.data.filters

    #cal plot lengths
    df['plot_word_count'] = df['Plot'].str.split().str.len()

    #filter by plot length
    df =df[df['plot_word_count'] >= filters.min_plot_length]
    if config.system.verbose:
        print(f"Filtered by plot length (>= {filters.min_plot_length} words)")

    #filter by year
    if 'Release Year' in df.columns:
        #convert to numeric
        df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce')

        #filter by year range
        df = df[df['Release Year'] >= filters.year_from]

        if filters.year_to is not None:
            df = df[df['Release Year'] <= filters.year_to]

        if config.system.verbose:
            year_range  = year_range = f"{filters.year_from} - {filters.year_to or 'present'}"
            print(f"Filtered by release year ({year_range})")

    return df

def get_dataset_stats(df: pd.DataFrame) -> dict:
    """get statistics about the dataset"""
    stats = {
        'total_movies': len(df),
        'avg_plot_length': df['Plot'].str.split().str.len().mean(),
        'min_plot_length': df['Plot'].str.split().str.len().min(),
        'max_plot_length': df['Plot'].str.split().str.len().max(),
    }

    if 'Release Year' in df.columns:
        stats['year_range'] = (
            int(df['Release Year'].min()),
            int(df['Release Year'].max())
        )

    if 'Genre' in df.columns:
        stats['unique_genres'] = df['Genre'].nunique()

    return stats

