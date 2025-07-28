from openai import OpenAI

"""
def get_embedding(text_to_embed):
    response = OpenAI.Embedding.create(
        model="text-embedding-ada-002",
        input=[text_to_embed]
    )
    
    embedding = response["data"][0]["embedding"]
    return embedding

text = "This is a sample text for embedding."
embedding_vector = get_embedding(text)
print(embedding_vector)
"""


class Asker:
    def __init__(self, model:str ="text-embedding-ada-002"):
        self.model = model
        self.openai_client = OpenAI()
        self.context = ""

        self.set_context("Allie is the wife of Tevon. She is a sales person. She enjoys playing guitar and crafting. Maddox is their son. He is 18 months old and likes to play with balls. Tevon is a mechanical engineer that enjoys soccer and aviation.")

    def ask(self, question: str):
        response = self.openai_client.responses.create(
            model=self.model,
            input=[
                {"role": "user", "content": question},
                {"role": "system", "content": f"You are a helpful assistant for answering question based on the provided context. Please provide a concise and accurate answer. Here is your context: {self.context}"}
            ]
        )
        
        return response.output_text
    
    def set_context(self, context: str):
        self.context = context