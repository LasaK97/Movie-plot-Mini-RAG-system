from typing import List, Dict, Any
import json
from pathlib import Path

def format_context(chunks: List[Dict], max_chars: int = 400) -> List[str]:
    """formats chunks to context strings for LLM"""
    contexts = []
    for chunk in chunks:
        text = chunk["text"]

        #truncate if long
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        #format with movie title
        context = f"[{chunk['movie_title']}] {text}"
        contexts.append(context)

    return contexts


def to_json_output(response: Dict[str, Any]) -> Dict[str, Any]:
    """ converts response to json string """
    return {
        "answer": response.get('answer', ''),
        "contexts": response.get('contexts', []),
        "reasoning": response.get('reasoning', '')
    }


def print_json_output(response: Dict[str, Any]) -> None:
    """ prints json output """
    json_output = to_json_output(response)
    print(json.dumps(json_output, indent=2, ensure_ascii=False))

def save_json(data: Any, filepath: Path) -> None:
    """Saves data to json file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filepath}")


def save_json_output(response: Dict[str, Any], filepath: Path) -> None:
    """saves json output """
    json_output = to_json_output(response)
    save_json(json_output, filepath)

def load_json(filepath: Path) -> Any:
    """loads data from json file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def truncate_text(text: str, max_length: int = 100) -> str:
    """truncates a string to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_results(response: Dict[str, Any]) -> None:
    """prints results"""
    print("\n" + "=" * 70)
    print("ANSWER".center(70))
    print("=" * 70)
    print(response.get('answer', 'No answer provided'))

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXTS".center(70))
    print("=" * 70)
    contexts = response.get('contexts', [])
    for i, context in enumerate(contexts, 1):
        print(f"\n[Context {i}]")
        print(context)

    print("\n" + "=" * 70)
    print("REASONING".center(70))
    print("=" * 70)
    print(response.get('reasoning', 'No reasoning provided'))

    # Print metadata if available
    if 'metadata' in response:
        metadata = response['metadata']
        print("\n" + "=" * 70)
        print("METADATA".center(70))
        print("=" * 70)
        print(f"Query time: {metadata.get('query_time', 0):.2f}s")
        print(f"Top similarity: {metadata.get('top_similarity', 0):.3f}")
        print(f"Contexts retrieved: {metadata.get('num_contexts', 0)}")

    print("=" * 70 + "\n")



