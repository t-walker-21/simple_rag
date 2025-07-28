import argparse
import yaml

from src.config.logging import logger
from src.rag_asker.asker import Asker

def main():
    parser = argparse.ArgumentParser(description="Simple RAG Application")
    parser.add_argument("--query", type=str, help="Ask your data a question")
    args = parser.parse_args()

    with open("src/config/llm.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        model = config.get("asker_model", "gpt-4.1")

    asker = Asker(model=model)

    if args.query:
        logger.info(f"Received query: {args.query}")

        response = asker.ask(args.query)
        logger.debug(f"Response generated: {response}")
        print(response)



if __name__ == "__main__":
    main()