from groq import Groq # type: ignore

class LLMModel:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def generate_response(self, model: str, prompt: str, max_tokens: int):
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
            )
            return completion
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
