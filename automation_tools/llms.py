import requests
import json
import os
from automation_tools.common import LLM

# Cortex API LLM implementation
class CortexLLM(LLM):    
    def __init__(self, settings):
        self.baseurl = settings.cortex_baseurl
        self.client_id = settings.cortex_client_id
        self.client_secret = settings.cortex_client_secret
        self.cert = settings.cortex_root_ca
        self.model = settings.llm_model
        
        # Prepare headers
        self.headers = {
            'x-lbg-client-id': self.client_id,
            'x-lbg-client-secret': self.client_secret
        }
        
        self.chat_url = f"{self.baseurl}/chat/completions"
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 400, **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 300
            },
            "stream": False,
            "logprobs": True
        }
        
        try:
            # Handle certificate verification - check if cert exists
            if self.cert and os.path.exists(self.cert):
                verify = self.cert
            else:
                verify = False
            
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=self.headers,
                verify=verify
            )
            response.raise_for_status()
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']
            return answer
        except Exception as e:
            raise Exception(f"Cortex API generation failed: {str(e)}")


# Factory
# Get LLM instance based on provider in settings
def get_llm(settings) -> LLM:
    provider = (settings.llm_provider or "cortex").lower()
    
    if provider in ["cortex", "vertex_ai"]:
        return CortexLLM(settings)
    
    raise ValueError(f"Unknown LLM provider: {provider}")
