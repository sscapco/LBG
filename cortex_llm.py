import json
import asyncio
from typing import List, Dict, Optional, Any, Union
import requests
import constants


class CortexLLMClient:
    """Synchronous wrapper for Cortex API that mimics OpenAI client interface"""
    
    def __init__(self):
        self.base_url = constants.BASEURL
        
        self.headers = {
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
        
        self.headers_with_content_type = {
            'Content-Type': 'application/json',
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
        
        self.default_model = "vertex_ai/gemini-2.5-flash"
        
        # OpenAI SDK compatibility attributes
        self.api_key = "cortex-placeholder-key"
        self.organization = None
        self.base_url_attr = self.base_url
    
    def __getstate__(self):
        """Make client pickle-safe for serialization"""
        return {
            'base_url': self.base_url,
            'default_model': self.default_model,
            'api_key': self.api_key,
            'organization': self.organization,
        }
    
    def __setstate__(self, state):
        """Restore client from pickle"""
        self.__init__()
    
    def _call_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 1000,
        response_format: Optional[Dict] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Internal method to call Cortex chat completions"""
        
        if stream:
            raise NotImplementedError("Streaming not implemented for Cortex client")
        
        chat_url = f"{self.base_url}/chat/completions"
        
        # Deep copy and clean messages
        clean_messages = []
        for msg in messages:
            if hasattr(msg, 'model_dump'):
                msg = msg.model_dump()
            elif hasattr(msg, 'dict'):
                msg = msg.dict()
            
            clean_messages.append({
                "role": str(msg.get("role", "user")),
                "content": str(msg.get("content", ""))
            })
        
        payload = {
            "model": model or self.default_model,
            "messages": clean_messages,
            "max_tokens": max_tokens,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 300
            },
            "stream": False,
            "logprobs": True
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        
        if response_format:
            if hasattr(response_format, 'model_dump'):
                response_format = response_format.model_dump()
            elif hasattr(response_format, 'dict'):
                response_format = response_format.dict()
            
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                if clean_messages and clean_messages[-1]["role"] == "user":
                    clean_messages[-1]["content"] += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations."
                    payload["messages"] = clean_messages
        
        try:
            response = requests.post(
                chat_url, 
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=60
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            return self._create_chat_completion(response_json)
            
        except Exception as e:
            print(f"❌ Error calling Cortex API: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_chat_completion(self, data: Dict) -> Any:
        """Create OpenAI-compatible ChatCompletion object"""
        
        class Message:
            def __init__(self, message_data):
                self.role = message_data.get("role", "assistant")
                self.content = message_data.get("content", "")
                self.tool_calls = message_data.get("tool_calls", None)
                self.function_call = message_data.get("function_call", None)
        
        class Choice:
            def __init__(self, choice_data):
                self.index = choice_data.get("index", 0)
                self.message = Message(choice_data.get("message", {}))
                self.finish_reason = choice_data.get("finish_reason", "stop")
        
        class ChatCompletion:
            def __init__(self, data):
                self.id = data.get("id", "cortex-completion")
                self.object = "chat.completion"
                self.created = data.get("created", 0)
                self.model = data.get("model", "vertex_ai/gemini-2.5-flash")
                self.usage = data.get("usage", {})
                
                if "choices" in data and isinstance(data["choices"], list):
                    self.choices = [Choice(choice) for choice in data["choices"]]
                else:
                    self.choices = [Choice({"message": {"content": str(data)}})]
        
        return ChatCompletion(data)
    
    @property
    def chat(self):
        return self
    
    @property
    def completions(self):
        return self
    
    def create(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ):
        return self._call_chat_completion(
            model=model or self.default_model,
            messages=messages or [],
            temperature=temperature,
            max_tokens=max_tokens or 1000,
            stream=stream or False,
            response_format=response_format,
            **kwargs
        )
    
    def _call_embeddings(self, model: str, input: Union[str, List[str]]) -> Any:
        
        embedding_url = f"{self.base_url}/embeddings"
        
        if isinstance(input, str):
            input = [input]
        
        embeddings_payload = {
            'model': model or 'vertex_ai/text-embedding-004',
            'input': input,
            'dimensions': 256,
            'encoding_format': 'float',
            'user': 'governance-pipeline'
        }
        
        try:
            embeddings_response = requests.post(
                embedding_url,
                json=embeddings_payload,
                headers=self.headers_with_content_type,
                verify=False,
                timeout=30
            )
            
            embeddings_response.raise_for_status()
            embedding_data = embeddings_response.json()
            
            return self._create_embedding_response(embedding_data)
            
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_embedding_response(self, data: Dict) -> Any:
        
        class Embedding:
            def __init__(self, item):
                self.object = "embedding"
                self.embedding = item.get("embedding", [])
                self.index = item.get("index", 0)
        
        class EmbeddingResponse:
            def __init__(self, data):
                self.object = "list"
                self.model = data.get("model", "vertex_ai/text-embedding-004")
                self.usage = data.get("usage", {})
                
                if "data" in data:
                    self.data = [Embedding(item) for item in data["data"]]
                else:
                    self.data = []
        
        return EmbeddingResponse(data)
    
    @property
    def embeddings(self):
        class EmbeddingsAPI:
            def __init__(self, parent):
                self.parent = parent
            
            def create(self, model: str, input: Union[str, List[str]], **kwargs):
                return self.parent._call_embeddings(model, input)
        
        return EmbeddingsAPI(self)


class AsyncCortexLLMClient:
    """Async wrapper - pickle-safe"""
    
    def __init__(self):
        self.sync_client = CortexLLMClient()
        
        self.base_url = self.sync_client.base_url
        self.default_model = self.sync_client.default_model
        self.api_key = self.sync_client.api_key
        self.organization = self.sync_client.organization
        self.base_url_attr = self.sync_client.base_url_attr
        self.timeout = None
        self.max_retries = 2
        self.default_headers = self.sync_client.headers
    
    def __getstate__(self):
        """Make async client pickle-safe - CRITICAL for Agents SDK"""
        return {
            'base_url': self.base_url,
            'default_model': self.default_model,
            'api_key': self.api_key,
            'organization': self.organization,
        }
    
    def __setstate__(self, state):
        """Restore async client from pickle"""
        self.__init__()
    
    @property
    def chat(self):
        return self
    
    @property
    def completions(self):
        return self
    
    async def create(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ):
        """Async create - calls sync directly"""
        result = self.sync_client._call_chat_completion(
            model=model or self.default_model,
            messages=messages or [],
            temperature=temperature if temperature is not None else 0,
            max_tokens=max_tokens or 1000,
            response_format=response_format,
            stream=stream or False,
            **kwargs
        )
        
        return result
    
    @property
    def embeddings(self):
        class AsyncEmbeddingsAPI:
            def __init__(self, parent):
                self.parent = parent
            
            async def create(self, model: str, input: Union[str, List[str]], **kwargs):
                return self.parent.sync_client._call_embeddings(model, input)
        
        return AsyncEmbeddingsAPI(self)
    
    def __getattr__(self, name):
        if hasattr(self.sync_client, name):
            return getattr(self.sync_client, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def get_llm_client(async_mode: bool = False):
    if async_mode:
        return AsyncCortexLLMClient()
    return CortexLLMClient()