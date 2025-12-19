
from automation_tools.common import LLM

class GeminiLLM(LLM):
    """Gemini LLM that uses cortex_connection"""
    
    def __init__(self, settings):
        # Import here to avoid circular imports
        try:
            from cortex_connection import cortex_client
            self.client = cortex_client
        except ImportError:
            # Fallback: create our own cortex client
            import sys
            import os
            # Add parent directory to path
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from cortex_connection import cortex_client
            self.client = cortex_client
        
        self.settings = settings

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Gemini API
        
        Args:
            prompt: The prompt text
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
        
        Returns:
            Generated text
        """
        temperature = kwargs.get('temperature', 0.0)
        max_tokens = kwargs.get('max_tokens', 500)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_enabled=False  # Disable for JSON responses
            )
            return response
        except Exception as e:
            print(f"Gemini LLM error: {e}")
            # Return empty JSON as fallback
            return "{}"


class AzureOpenAI(LLM):
    """Legacy Azure OpenAI support (kept for compatibility)"""
    
    def __init__(self, settings):
        from openai import AzureOpenAI as AzureClient
        self.client = AzureClient(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint
        )
        self.deployment = settings.azure_deployment

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get('temperature', 0.0)
        max_tokens = kwargs.get('max_tokens', 4096)
        
        res = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            model=self.deployment,
            temperature=temperature   
        )
        return res.choices[0].message.content


# Factory
def get_llm(settings) -> LLM:
    """
    Get LLM instance based on settings
    
    Supports:
    - gemini: Uses Gemini via cortex_connection (default)
    - azure_openai: Uses Azure OpenAI (legacy)
    """
    provider = (settings.llm_provider or "gemini").lower()

    if provider in ["gemini", "vertex_ai", "cortex"]:
        return GeminiLLM(settings)
    
    if provider == "azure_openai":
        return AzureOpenAI(settings)

    # Default to Gemini if unknown
    print(f"Warning: Unknown LLM provider '{provider}', defaulting to Gemini")
    return GeminiLLM(settings)