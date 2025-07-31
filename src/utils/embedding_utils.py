from src.utils.text_utils import doc_to_chunks
import json

def file_to_chunks(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Reads a file and converts its content into smaller chunks.
    
    Args:
        file_path (str): The path to the file to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of characters to overlap between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    return doc_to_chunks(content, chunk_size, overlap)

def embeddings_to_file(embeddings: list, chunks: list, file_path: str):
    """
    Saves embeddings and their corresponding chunks to a np json file.

    Args:
        embeddings (list): The list of embeddings to save.
        chunks (list): The list of text chunks corresponding to the embeddings.
        file_path (str): The path to the file where embeddings will be saved.
    """
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump({"embeddings": embeddings, "chunks": chunks}, file)

def load_embeddings_from_file(file_path: str) -> dict:
    """
    Loads embeddings and their corresponding chunks from a file.
    Args:
        file_path (str): The path to the file containing embeddings.
    
    Returns:
        dict: A dictionary containing embeddings and their corresponding chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)