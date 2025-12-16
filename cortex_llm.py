import json
import asyncio
from typing import List, Dict, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import requests
import constants


class CortexLLMClient:
    """Synchronous wrapper for Cortex API that mimics OpenAI client interface"""
    
    def __init__(self):
        self.base_url = constants.BASEURL
        
        # Headers matching your working Cortex_Connection.py
        self.headers = {
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
        
        # Headers for embeddings (with Content-Type)
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
        
        # Payload matching your working Cortex_Connection.py
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 300
            },
            "stream": False,
            "logprobs": True
        }
        
        # Add temperature if specified
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Handle JSON response format request
        if response_format and response_format.get("type") == "json_object":
            # Append instruction to last user message
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations."
        
        try:
            # Use requests.post directly with json=payload (matching your working code)
            response = requests.post(
                chat_url, 
                json=payload,  # Use json= not data=json.dumps()
                headers=self.headers,
                verify=False  # Matching your working code
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            # Create OpenAI-compatible response
            return self._create_chat_completion(response_json)
            
        except Exception as e:
            print(f"❌ Error calling Cortex API: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_chat_completion(self, data: Dict) -> Any:
        """Create OpenAI-compatible ChatCompletion object from Cortex response"""
        
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
                
                # Parse choices from Cortex response
                if "choices" in data and isinstance(data["choices"], list):
                    self.choices = [Choice(choice) for choice in data["choices"]]
                else:
                    # Fallback
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
        """Sync create method"""
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
        """Internal method to call Cortex embeddings - matching your working code"""
        
        embedding_url = f"{self.base_url}/embeddings"
        
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
        
        # Payload matching your working Cortex_Connection.py
        embeddings_payload = {
            'model': model or 'vertex_ai/text-embedding-004',
            'input': input,
            'dimensions': 256,
            'encoding_format': 'float',
            'user': 'user-1223345'
        }
        
        try:
            # Use requests.post directly (matching your working code)
            embeddings_response = requests.post(
                embedding_url,
                json=embeddings_payload,  # Use json= not data=json.dumps()
                headers=self.headers_with_content_type,
                verify=False  # Matching your working code
            )
            
            embeddings_response.raise_for_status()
            embedding_data = embeddings_response.json()
            
            # Create OpenAI-compatible response
            return self._create_embedding_response(embedding_data)
            
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_embedding_response(self, data: Dict) -> Any:
        """Create OpenAI-compatible EmbeddingResponse object"""
        
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
                
                # Parse embeddings from Cortex response
                if "data" in data:
                    self.data = [Embedding(item) for item in data["data"]]
                else:
                    self.data = []
        
        return EmbeddingResponse(data)
    
    @property
    def embeddings(self):
        return self
    
    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs):
        return self._call_embeddings(model, input)


class AsyncCortexLLMClient:
    """Async wrapper for Cortex API using ThreadPoolExecutor"""
    
    def __init__(self):
        self.sync_client = CortexLLMClient()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Copy attributes for OpenAI SDK compatibility
        self.base_url = self.sync_client.base_url
        self.default_model = self.sync_client.default_model
        self.api_key = self.sync_client.api_key
        self.organization = self.sync_client.organization
        self.base_url_attr = self.sync_client.base_url_attr
        self.timeout = None
        self.max_retries = 2
        self.default_headers = self.sync_client.headers
    
    async def _async_call(self, func, *args, **kwargs):
        """Generic async wrapper"""
        loop = asyncio.get_event_loop()
        
        def _call():
            return func(*args, **kwargs)
        
        result = await loop.run_in_executor(self.executor, _call)
        return result
    
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
        """Async create method"""
        result = await self._async_call(
            self.sync_client._call_chat_completion,
            model or self.default_model,
            messages or [],
            temperature,
            max_tokens or 1000,
            response_format,
            stream or False,
            **kwargs
        )
        return result
    
    @property
    def embeddings(self):
        return self
    
    async def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs):
        """Async embeddings"""
        return await self._async_call(
            self.sync_client._call_embeddings,
            model,
            input
        )
    
    def __getattr__(self, name):
        """Fallback for unexpected attributes"""
        if hasattr(self.sync_client, name):
            attr = getattr(self.sync_client, name)
            if callable(attr):
                async def async_wrapper(*args, **kwargs):
                    return await self._async_call(attr, *args, **kwargs)
                return async_wrapper
            return attr
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def get_llm_client(async_mode: bool = False):
    """Factory function"""
    if async_mode:
        return AsyncCortexLLMClient()
    return CortexLLMClient()