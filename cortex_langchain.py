"""
LangChain-compatible wrapper for Cortex API
This allows us to use Cortex with LangGraph
"""
from typing import List, Dict, Any, Optional, Iterator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
import requests
import constants


class CortexChatModel(BaseChatModel):
    """LangChain-compatible Cortex chat model"""
    
    model_name: str = "vertex_ai/gemini-2.5-flash"
    temperature: float = 0
    max_tokens: int = 1000
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = constants.BASEURL
        self.headers = {
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
    
    @property
    def _llm_type(self) -> str:
        return "cortex"
    
    def _convert_messages_to_cortex_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert LangChain messages to Cortex format"""
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
        
        # Call Cortex API
        chat_url = f"{self.base_url}/chat/completions"
        
        try:
            response = requests.post(
                chat_url,
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=60
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract content
            content = response_data["choices"][0]["message"]["content"]
            
            # Create LangChain response
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            print(f"❌ Error calling Cortex: {e}")
            raise
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version - just calls sync for now since requests is blocking anyway"""
        return self._generate(messages, stop, run_manager, **kwargs)


class CortexEmbeddings:
    """LangChain-compatible Cortex embeddings"""
    
    def __init__(self):
        self.base_url = constants.BASEURL
        self.headers = {
            'Content-Type': 'application/json',
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
        self.model = 'vertex_ai/text-embedding-004'
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding_url = f"{self.base_url}/embeddings"
        
        payload = {
            'model': self.model,
            'input': [text],
            'dimensions': 256,
            'encoding_format': 'float',
            'user': 'governance-pipeline'
        }
        
        try:
            response = requests.post(
                embedding_url,
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            return data['data'][0]['embedding']
            
        except Exception as e:
            print(f"❌ Embeddings error: {e}")
            raise