import argparse
import yaml
import numpy as np
import time

from src.config.logging import logger
from src.rag_asker.asker import Asker
from src.rag_embedder.embedder import Embedder
from src.utils.embedding_utils import file_to_chunks, embeddings_to_file, load_embeddings_from_file

def main():
    parser = argparse.ArgumentParser(description="Simple RAG Application")
    parser.add_argument("--query", type=str, help="Ask your data a question")
    parser.add_argument("--embed", type=str, help="Embed a text passage. This is path to the text file to embed")
    parser.add_argument("--embeddings", type=str, help="Path to the embeddings file to load or save embeddings", default="embeddings.json")
    args = parser.parse_args()

    with open("src/config/llm.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        asker_model_name = config.get("asker_model", "gpt-4.1")
        embedder_model_name = config.get("embedder_model", "text-embedding-ada-002")

    asker = Asker(model_name=asker_model_name)
    embedder = Embedder(model_name=embedder_model_name)

    if args.query:
        logger.info(f"Received query: {args.query}")

        corpus_data = load_embeddings_from_file(args.embeddings)

        corpus_embeddings = corpus_data.get("embeddings", [])
        chunks = corpus_data.get("chunks", [])

        query_embedding = np.array(embedder.embed(args.query))

        chunk_scores = []

        start_time = time.time()
        for embedding, chunk in zip(corpus_embeddings, chunks):
            embedding = np.array(embedding)
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            chunk_scores.append((similarity, chunk))

        # Get top 3 most similar chunks
        chunk_scores = sorted(chunk_scores, key=lambda x: x[0], reverse=True)[:3]
        end_time = time.time()
        logger.info(f"Time taken for similarity computation: {end_time - start_time} seconds")

        asker.set_context(" ".join([chunk for _, chunk in chunk_scores]))
        response, context = asker.ask(args.query)
        logger.debug(f"Response generated: {response}")
        print(response)
        print(f"Context used: {context}")

    if args.embed:
        logger.info(f"Received embed request: {args.embed}")

        chunks = file_to_chunks(args.embed, chunk_size=300, overlap=100)        
        embeddings = [embedder.embed(chunk) for chunk in chunks]

        embeddings_to_file(embeddings, chunks, args.embeddings)

if __name__ == "__main__":
    main()