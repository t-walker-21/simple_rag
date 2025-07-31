# Utils for chunking text into smaller parts
from typing import List

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits the input text into smaller chunks of a specified size and overlap.
    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of characters to overlap between chunks.
    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def doc_to_chunks(doc: str, chunk_size: int = 5, overlap: int = 1) -> List[str]:
    """
    Converts a document into smaller chunks.
    Args:
        doc (str): The document to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of characters to overlap between chunks.
    Returns:
        List[str]: A list of text chunks.
    """
    return chunk_text(doc, chunk_size, overlap)