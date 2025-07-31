from openai import OpenAI

class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.openai_client = OpenAI()

    def embed(self, text: str) -> list:
        response = self.openai_client.embeddings.create(
            model=self.model_name,
            input=text
        )

        return (response.data[0].embedding if response.data else None)