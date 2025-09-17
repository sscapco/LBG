from src.adapters.common import Embeddings

class AzureOpenAIEmbeddings(Embeddings):
    def __init__(self, settings):
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint
        )
        # Default to text-embedding-3-large unless overridden
        self.deployment = settings.azure_embedding_deployment or "text-embedding-3-large"

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(
            input=text,
            model=self.deployment
        )
        return resp.data[0].embedding


def get_embeddings(settings) -> Embeddings:
    provider = (settings.embedding_provider or "").lower()
    if provider == "azure_openai":
        return AzureOpenAIEmbeddings(settings)
    raise ValueError(f"Unknown embedding provider: {provider}")
