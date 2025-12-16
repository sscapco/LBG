import json
import asyncio
from typing import List, Dict, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import requests
from api_requests import apihandler
import constants


class CortexLLMClient:
    """Synchronous wrapper for Cortex API that mimics OpenAI client interface"""
    
    def __init__(self):
        self.api = apihandler()
        self.base_url = constants.BASEURL
        self.headers = {
            'Content-Type': 'application/json',
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
        self.cert = constants.Root_CA
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
        
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        
        # Add response format if JSON requested
        if response_format and response_format.get("type") == "json_object":
            modified_messages = [msg.copy() for msg in messages]
            if modified_messages:
                last_msg = modified_messages[-1]
                if last_msg["role"] == "user":
                    last_msg["content"] += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations."
                else:
                    modified_messages.append({
                        "role": "user",
                        "content": "Respond ONLY with valid JSON."
                    })
            payload["messages"] = modified_messages
        
        try:
            print(f"🔧 Calling Cortex API...")
            
            # Call the API
            api_response = self.api.call_api_post(
                url=chat_url,
                headers=self.headers,
                payload=payload,
                cert=self.cert
            )
            
            print(f"🔍 API Response Type: {type(api_response)}")
            
            # Handle None response
            if api_response is None:
                error_msg = "API call returned None - check your api_requests.py for errors"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            # Parse response
            response_data = self._parse_api_response(api_response)
            
            # Create and return ChatCompletion object
            completion = self._create_chat_completion(response_data)
            
            print(f"✅ Completion created successfully")
            return completion
            
        except Exception as e:
            print(f"❌ Error in _call_chat_completion: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _parse_api_response(self, api_response: Any) -> Dict:
        """Parse various API response formats into a standard dict"""
        
        response_data = None
        
        # Case 1: Tuple (response, status_code)
        if isinstance(api_response, tuple):
            print(f"📦 Tuple with {len(api_response)} elements")
            actual_response = api_response[0]
            
            if hasattr(actual_response, 'json'):
                response_data = actual_response.json()
            elif hasattr(actual_response, 'text'):
                response_data = json.loads(actual_response.text)
            elif isinstance(actual_response, dict):
                response_data = actual_response
            elif isinstance(actual_response, str):
                try:
                    response_data = json.loads(actual_response)
                except:
                    response_data = {"choices": [{"message": {"content": actual_response}}]}
            else:
                response_data = {"choices": [{"message": {"content": str(actual_response)}}]}
        
        # Case 2: Response object with .json()
        elif hasattr(api_response, 'json'):
            response_data = api_response.json()
            print(f"✅ Parsed from response.json()")
        
        # Case 3: Already a dict
        elif isinstance(api_response, dict):
            response_data = api_response
            print(f"✅ Already a dict")
        
        # Case 4: String
        elif isinstance(api_response, str):
            try:
                response_data = json.loads(api_response)
                print(f"✅ Parsed from JSON string")
            except:
                response_data = {"choices": [{"message": {"content": api_response}}]}
                print(f"⚠️ Created fallback from string")
        
        # Case 5: Unknown
        else:
            print(f"⚠️ Unknown type: {type(api_response)}")
            response_data = {"choices": [{"message": {"content": str(api_response)}}]}
        
        # Validate structure
        if isinstance(response_data, dict):
            print(f"📊 Keys: {list(response_data.keys())}")
            if "choices" in response_data:
                print(f"   ✓ Has {len(response_data['choices'])} choices")
        
        return response_data
    
    def _create_chat_completion(self, data: Dict) -> Any:
        """Create OpenAI-compatible ChatCompletion object"""
        
        # Define inner classes at proper scope
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
                
                # Create choices
                if "choices" in data and isinstance(data["choices"], list):
                    self.choices = [Choice(choice) for choice in data["choices"]]
                else:
                    print("⚠️ No choices array, creating fallback")
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
            temperature=temperature if temperature is not None else 0,
            max_tokens=max_tokens or 1000,
            stream=stream or False,
            response_format=response_format,
            **kwargs
        )
    
    def _call_embeddings(self, model: str, input: Union[str, List[str]]) -> Any:
        """Internal method to call Cortex embeddings"""
        
        embedding_url = f"{self.base_url}/embeddings"
        
        if isinstance(input, str):
            input = [input]
        
        payload = {
            'model': model or 'vertex_ai/text-embedding-004',
            'input': input,
            'dimensions': 256,
            'encoding_format': 'float',
            'user': 'governance-pipeline'
        }
        
        try:
            response = requests.post(
                embedding_url,
                json=payload,
                headers=self.headers,
                verify=self.cert
            )
            
            response_data = response.json()
            return self._create_embedding_response(response_data)
            
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
                self.data = [Embedding(item) for item in data.get("data", [])]
        
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
        """Generic async wrapper - FIXED VERSION"""
        loop = asyncio.get_event_loop()
        
        # Create a wrapper function that calls the sync function with unpacked args
        def _call():
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"❌ Error in async wrapper: {e}")
                raise
        
        # Run in executor
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
        """Async create method - FIXED"""
        print(f"🔄 AsyncCortexLLMClient.create() called")
        
        # Call the sync client's method with proper arguments
        result = await self._async_call(
            self.sync_client._call_chat_completion,
            model or self.default_model,
            messages or [],
            temperature if temperature is not None else 0,
            max_tokens or 1000,
            response_format,
            stream or False,
            **kwargs
        )
        
        print(f"✅ AsyncCortexLLMClient.create() returning: {type(result)}")
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
        
        print(f"⚠️ Missing attribute: {name}")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def get_llm_client(async_mode: bool = False):
    """Factory function"""
    if async_mode:
        return AsyncCortexLLMClient()
    return CortexLLMClient()