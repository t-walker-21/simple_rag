from openai import OpenAI

class Asker:
    def __init__(self, model:str):
        self.model = model
        self.openai_client = OpenAI()
        self.context = ""

    def ask(self, question: str):
        response = self.openai_client.responses.create(
            model=self.model,
            input=[
                {"role": "user", "content": question},
                {"role": "system", "content": f"You are a helpful assistant for answering question based on the provided context. Please provide a concise and accurate answer, and if the context is not sufficient, say so and ask for more context. Here is your context: {self.context}"}
            ]
        )
        
        return response.output_text, self.context
    
    def set_context(self, context: str):
        self.context = context
