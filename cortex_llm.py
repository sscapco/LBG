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
        self.cert = constants.root_CA
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
            # Modify the last message to request JSON output
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
            # Call the API
            api_response = self.api.call_api_post(
                url=chat_url,
                headers=self.headers,
                payload=payload,
                cert=self.cert
            )
            
            # Handle different response types from call_api_post
            response_data = None
            
            # Case 1: Tuple returned (response, status_code) or similar
            if isinstance(api_response, tuple):
                print(f"📦 API returned tuple with {len(api_response)} elements")
                # Usually the first element is the actual response
                actual_response = api_response[0]
                
                if hasattr(actual_response, 'json'):
                    response_data = actual_response.json()
                elif isinstance(actual_response, dict):
                    response_data = actual_response
                elif isinstance(actual_response, str):
                    try:
                        response_data = json.loads(actual_response)
                    except:
                        response_data = {"choices": [{"message": {"content": actual_response}}]}
                else:
                    response_data = {"choices": [{"message": {"content": str(actual_response)}}]}
            
            # Case 2: Response object with .json() method
            elif hasattr(api_response, 'json'):
                response_data = api_response.json()
            
            # Case 3: Already a dict
            elif isinstance(api_response, dict):
                response_data = api_response
            
            # Case 4: String (try to parse as JSON)
            elif isinstance(api_response, str):
                try:
                    response_data = json.loads(api_response)
                except:
                    response_data = {"choices": [{"message": {"content": api_response}}]}
            
            # Case 5: Unknown type
            else:
                print(f"⚠️ Unexpected response type: {type(api_response)}")
                response_data = {"choices": [{"message": {"content": str(api_response)}}]}
            
            # Debug: Print response structure
            print(f"📊 Response data keys: {response_data.keys() if isinstance(response_data, dict) else 'not a dict'}")
            
            # Create OpenAI-compatible response object
            return self._create_chat_completion(response_data)
            
        except Exception as e:
            print(f"❌ Error calling Cortex API: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_chat_completion(self, data: Dict) -> Any:
        """Create OpenAI-compatible ChatCompletion object"""
        class ChatCompletion:
            def __init__(self, data):
                self.id = data.get("id", "cortex-completion")
                self.object = "chat.completion"
                self.created = data.get("created", 0)
                self.model = data.get("model", "vertex_ai/gemini-2.5-flash")
                self.choices = self._create_choices(data)
                self.usage = data.get("usage", {})
                
                # Debug
                print(f"✅ Created ChatCompletion with {len(self.choices)} choices")
            
            def _create_choices(self, data):
                if "choices" in data and isinstance(data["choices"], list):
                    return [self._create_choice(choice) for choice in data["choices"]]
                else:
                    # Fallback: create a single choice from the entire data
                    print("⚠️ No 'choices' array found, creating fallback choice")
                    return [self._create_choice({"message": {"content": str(data)}})]
            
            def _create_choice(self, choice_data):
                class Choice:
                    def __init__(self, choice_data):
                        self.index = choice_data.get("index", 0)
                        self.message = self._create_message(choice_data.get("message", {}))
                        self.finish_reason = choice_data.get("finish_reason", "stop")
                    
                    def _create_message(self, message_data):
                        class Message:
                            def __init__(self, message_data):
                                self.role = message_data.get("role", "assistant")
                                self.content = message_data.get("content", "")
                                self.tool_calls = message_data.get("tool_calls", None)
                                self.function_call = message_data.get("function_call", None)
                                
                                # Debug
                                if self.content:
                                    print(f"💬 Message content length: {len(self.content)}")
                        return Message(message_data)
                return Choice(choice_data)
        
        return ChatCompletion(data)
    
    @property
    def chat(self):
        """Property to match OpenAI client.chat interface"""
        return self
    
    @property
    def completions(self):
        """Property to match OpenAI client.chat.completions interface"""
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
        """Method to match OpenAI client.chat.completions.create interface"""
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
        
        # Ensure input is a list
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
            # Direct requests call (not using apihandler for embeddings)
            response = requests.post(
                embedding_url,
                json=payload,
                headers=self.headers,
                verify=self.cert
            )
            
            response_data = response.json()
            
            # Create OpenAI-compatible response
            return self._create_embedding_response(response_data)
            
        except Exception as e:
            print(f"❌ Error calling Cortex Embeddings API: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_embedding_response(self, data: Dict) -> Any:
        """Create OpenAI-compatible EmbeddingResponse object"""
        class EmbeddingResponse:
            def __init__(self, data):
                self.object = "list"
                self.data = self._create_embeddings(data)
                self.model = data.get("model", "vertex_ai/text-embedding-004")
                self.usage = data.get("usage", {})
            
            def _create_embeddings(self, data):
                if "data" in data:
                    return [self._create_embedding(item) for item in data["data"]]
                else:
                    # Fallback
                    return []
            
            def _create_embedding(self, item):
                class Embedding:
                    def __init__(self, item):
                        self.object = "embedding"
                        self.embedding = item.get("embedding", [])
                        self.index = item.get("index", 0)
                return Embedding(item)
        
        return EmbeddingResponse(data)
    
    @property
    def embeddings(self):
        """Property to match OpenAI client.embeddings interface"""
        return self
    
    def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs):
        """Method to match OpenAI embeddings.create interface"""
        return self._call_embeddings(model, input)


class AsyncCortexLLMClient:
    """Async wrapper for Cortex API using ThreadPoolExecutor"""
    
    def __init__(self):
        self.sync_client = CortexLLMClient()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Copy essential attributes from sync client for OpenAI SDK compatibility
        self.base_url = self.sync_client.base_url
        self.default_model = self.sync_client.default_model
        
        # CRITICAL: OpenAI Agents SDK requires these attributes
        self.api_key = self.sync_client.api_key
        self.organization = self.sync_client.organization
        self.base_url_attr = self.sync_client.base_url_attr
        
        # Additional attributes that might be checked
        self.timeout = None
        self.max_retries = 2
        self.default_headers = self.sync_client.headers
    
    async def _async_call(self, func, *args, **kwargs):
        """Generic async wrapper for sync functions"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            lambda: func(*args, **kwargs)
        )
    
    @property
    def chat(self):
        """Return self to allow client.chat.completions.create() chaining"""
        return self
    
    @property
    def completions(self):
        """Return self to allow client.chat.completions.create() chaining"""
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
        """Async version of chat completions create"""
        return await self._async_call(
            self.sync_client._call_chat_completion,
            model=model or self.default_model,
            messages=messages or [],
            temperature=temperature if temperature is not None else 0,
            max_tokens=max_tokens or 1000,
            stream=stream or False,
            response_format=response_format,
            **kwargs
        )
    
    @property
    def embeddings(self):
        """Return self to allow client.embeddings.create() chaining"""
        return self
    
    async def create_embedding(self, model: str, input: Union[str, List[str]], **kwargs):
        """Async version of embeddings create"""
        return await self._async_call(
            self.sync_client._call_embeddings,
            model,
            input
        )
    
    def __getattr__(self, name):
        """
        Fallback for any unexpected attribute access.
        This helps debug what the Agents SDK is looking for.
        """
        # Try to get from sync_client first
        if hasattr(self.sync_client, name):
            attr = getattr(self.sync_client, name)
            # If it's a callable, wrap it in async
            if callable(attr):
                async def async_wrapper(*args, **kwargs):
                    return await self._async_call(attr, *args, **kwargs)
                return async_wrapper
            return attr
        
        # If not found, raise informative error
        print(f"⚠️ AsyncCortexLLMClient: Unexpected attribute access: {name}")
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"The OpenAI Agents SDK may be looking for this attribute."
        )


# Factory function to get the right client
def get_llm_client(async_mode: bool = False):
    """
    Factory function to get sync or async Cortex client
    
    Args:
        async_mode: If True, returns AsyncCortexLLMClient, else CortexLLMClient
    
    Returns:
        Client compatible with OpenAI API interface
    """
    if async_mode:
        return AsyncCortexLLMClient()
    return CortexLLMClient()