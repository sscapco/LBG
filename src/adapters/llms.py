from src.adapters.common import LLM

class AzureOpenAI(LLM):
    def __init__(self, settings):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint
        )
        self.deployment = settings.azure_deployment
        
        

    def generate(self, prompt: str, **kwargs) -> str:
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            model=self.deployment,
            temperature=0   
            )
        return res.choices[0].message.content


# Factory
def get_llm(settings) -> LLM:
    provider = (settings.llm_provider or "").lower()

    if provider == "azure_openai":
        return AzureOpenAI(settings)

    raise ValueError(f"Unknown LLM provider: {provider}")
