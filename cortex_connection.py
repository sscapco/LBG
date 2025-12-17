import requests
import json
from typing import List, Dict, Any, Optional
import constants
from api_requests import APIHandler


class CortexConnection:
    """Handler for Gemini API connections through Cortex"""
    
    def __init__(self):
        self.api = APIHandler()
        
        # Headers for chat completions
        self.headers = {
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
        
        # Headers for embeddings (with Content-Type)
        self.headers_embeddings = {
            'Content-Type': 'application/json',
            'x-lbg-client-id': constants.CLIENT_ID,
            'x-lbg-client-secret': constants.CLIENT_SECRET
        }
        
        # API endpoints
        self.chat_url = f"{constants.BASEURL}/chat/completions"
        self.embedding_url = f"{constants.BASEURL}/embeddings"
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = constants.MAX_TOKENS,
        temperature: float = 0.0,
        thinking_enabled: bool = True
    ) -> str:
        """
        Get chat completion from Gemini model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 for deterministic)
            thinking_enabled: Whether to enable thinking budget
            
        Returns:
            The model's response content as string
        """
        payload = {
            "model": constants.CHAT_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "logprobs": True
        }
        
        if thinking_enabled:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": constants.THINKING_BUDGET_TOKENS
            }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=self.headers,
                verify=False
            )
            response.raise_for_status()
            
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']
            return answer
            
        except Exception as e:
            print(f"Chat completion error: {e}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        # Clean text
        text = (text or "").replace("\n", " ")
        
        embeddings_payload = {
            'model': constants.EMBEDDING_MODEL,
            'input': [text],
            'dimensions': constants.EMBEDDING_DIMENSIONS,
            'encoding_format': 'float',
            'user': 'user-1223345'
        }
        
        try:
            embeddings_response = requests.post(
                self.embedding_url,
                json=embeddings_payload,
                headers=self.headers_embeddings,
                verify=False
            )
            embeddings_response.raise_for_status()
            
            embedding_data = embeddings_response.json()
            embedding_vector = embedding_data['data'][0]['embedding']
            return embedding_vector
            
        except Exception as e:
            print(f"Embedding error: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a batch
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Clean texts
        cleaned_texts = [(text or "").replace("\n", " ") for text in texts]
        
        embeddings_payload = {
            'model': constants.EMBEDDING_MODEL,
            'input': cleaned_texts,
            'dimensions': constants.EMBEDDING_DIMENSIONS,
            'encoding_format': 'float',
            'user': 'user-1223345'
        }
        
        try:
            embeddings_response = requests.post(
                self.embedding_url,
                json=embeddings_payload,
                headers=self.headers_embeddings,
                verify=False
            )
            embeddings_response.raise_for_status()
            
            embedding_data = embeddings_response.json()
            embeddings = [item['embedding'] for item in embedding_data['data']]
            return embeddings
            
        except Exception as e:
            print(f"Batch embedding error: {e}")
            raise


# Global instance for easy access
cortex_client = CortexConnection()