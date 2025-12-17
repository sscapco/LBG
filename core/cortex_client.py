"""
LangChain-compatible Cortex API client.
"""
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
import requests
import json

from config import CortexConfig


class CortexChatModel(BaseChatModel):
    """
    LangChain-compatible chat model for Cortex API.
    
    This allows LangGraph to use Cortex (Vertex AI/Gemini) as the LLM backend.
    """
    
    model_name: str = CortexConfig.CHAT_MODEL
    temperature: float = 0.0
    max_tokens: int = 1000
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = CortexConfig.BASE_URL
        self.headers = {
            'x-lbg-client-id': CortexConfig.CLIENT_ID,
            'x-lbg-client-secret': CortexConfig.CLIENT_SECRET
        }
        self.timeout = CortexConfig.CHAT_TIMEOUT
    
    @property
    def _llm_type(self) -> str:
        return "cortex"
    
    def _convert_messages_to_cortex_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages to Cortex API format"""
        cortex_messages = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "user"
            
            cortex_messages.append({
                "role": role,
                "content": str(msg.content)
            })
        
        return cortex_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from Cortex API"""
        
        # Convert messages
        cortex_messages = self._convert_messages_to_cortex_format(messages)
        
        # Build payload
        payload = {
            "model": self.model_name,
            "messages": cortex_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "thinking": {
                "type": "enabled",
                "budget_tokens": 300
            },
            "stream": False,
            "logprobs": True
        }
        
        # Handle JSON response format
        if kwargs.get("response_format") == "json":
            if cortex_messages and cortex_messages[-1]["role"] == "user":
                cortex_messages[-1]["content"] += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations."
                payload["messages"] = cortex_messages
        
        # Call Cortex API
        chat_url = f"{self.base_url}/chat/completions"
        
        try:
            response = requests.post(
                chat_url,
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract content
            content = response_data["choices"][0]["message"]["content"]
            
            # Create LangChain response
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except requests.exceptions.Timeout:
            raise Exception(f"Cortex API timeout after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Cortex API HTTP error: {e}")
        except Exception as e:
            raise Exception(f"Cortex API error: {e}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version - calls sync for now since requests is blocking"""
        return self._generate(messages, stop, run_manager, **kwargs)


class CortexEmbeddings:
    """
    Cortex embeddings client compatible with LangChain patterns.
    """
    
    def __init__(self):
        self.base_url = CortexConfig.BASE_URL
        self.headers = {
            'Content-Type': 'application/json',
            'x-lbg-client-id': CortexConfig.CLIENT_ID,
            'x-lbg-client-secret': CortexConfig.CLIENT_SECRET
        }
        self.model = CortexConfig.EMBEDDING_MODEL
        self.dimensions = CortexConfig.EMBEDDING_DIMENSIONS
        self.timeout = CortexConfig.EMBEDDING_TIMEOUT
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding_url = f"{self.base_url}/embeddings"
        
        payload = {
            'model': self.model,
            'input': [text],
            'dimensions': self.dimensions,
            'encoding_format': 'float',
            'user': 'governance-pipeline'
        }
        
        try:
            response = requests.post(
                embedding_url,
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data['data'][0]['embedding']
            
        except requests.exceptions.Timeout:
            raise Exception(f"Embeddings API timeout after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Embeddings API HTTP error: {e}")
        except Exception as e:
            raise Exception(f"Embeddings API error: {e}")


def get_cortex_llm(temperature: float = 0.0, max_tokens: int = 1000) -> CortexChatModel:
    """Factory function to create Cortex LLM"""
    return CortexChatModel(temperature=temperature, max_tokens=max_tokens)


def get_cortex_embeddings() -> CortexEmbeddings:
    """Factory function to create Cortex embeddings client"""
    return CortexEmbeddings()
