"""
Cortex API connection handler for chat completions and embeddings
"""
import requests
import json
from typing import List, Dict, Any, Optional
from api_requests import APIHandler
import constants


class CortexConnection:
    """Handler for Cortex API chat and embedding operations"""
    
    def __init__(self):
        self.api = APIHandler()
        self.base_url = constants.BASEURL
        self.client_id = constants.CLIENT_ID
        self.client_secret = constants.CLIENT_SECRET
        self.cert = constants.ROOT_CA
        
        # Standard headers for chat/embeddings
        self.headers = {
            'x-lbg-client-id': self.client_id,
            'x-lbg-client-secret': self.client_secret
        }
        
        # Headers for client credentials
        self.headers_credentials = {
            'Content-Type': 'application/json',
            'x-lbg-client-id': self.client_id,
            'x-lbg-client-secret': self.client_secret
        }
        
        self.chat_url = f"{self.base_url}/chat/completions"
        self.embedding_url = f"{self.base_url}/embeddings"
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        thinking_enabled: bool = True,
        thinking_budget: int = None,
        stream: bool = False,
        logprobs: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using Cortex API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (defaults to constants.CHAT_MODEL)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            thinking_enabled: Enable thinking mode
            thinking_budget: Thinking budget tokens (defaults to constants.THINKING_BUDGET_TOKENS)
            stream: Enable streaming
            logprobs: Return log probabilities
            
        Returns:
            Dict containing the API response
        """
        model = model or constants.CHAT_MODEL
        thinking_budget = thinking_budget or constants.THINKING_BUDGET_TOKENS
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "logprobs": logprobs
        }
        
        if thinking_enabled:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
        
        try:
            verify = self.cert if self.cert else False
            response = requests.post(
                self.chat_url,
                json=payload,
                headers=self.headers,
                verify=verify
            )
            response.raise_for_status()
            response_json = response.json()
            return response_json
        except requests.exceptions.RequestException as e:
            raise Exception(f"Chat completion failed: {str(e)}")
    
    def get_chat_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Convenience method to get just the text response from chat completion
        
        Args:
            messages: List of message dicts
            **kwargs: Additional arguments passed to chat_completion
            
        Returns:
            The text content of the response
        """
        response_json = self.chat_completion(messages, **kwargs)
        answer = response_json['choices'][0]['message']['content']
        return answer
    
    def get_embedding(
        self,
        text: str,
        model: str = None,
        dimensions: int = None
    ) -> List[float]:
        """
        Get embedding vector for a single text
        
        Args:
            text: Input text to embed
            model: Embedding model (defaults to constants.EMBEDDING_MODEL)
            dimensions: Output dimensions (defaults to constants.EMBEDDING_DIMENSIONS)
            
        Returns:
            List of floats representing the embedding vector
        """
        model = model or constants.EMBEDDING_MODEL
        dimensions = dimensions or constants.EMBEDDING_DIMENSIONS
        
        # Clean text
        text = (text or "").replace("\n", " ")
        
        payload = {
            'model': model,
            'input': [text],
            'dimensions': dimensions,
            'encoding_format': 'float',
            'user': 'governance-agent'
        }
        
        try:
            verify = self.cert if self.cert else False
            response = requests.post(
                self.embedding_url,
                json=payload,
                headers=self.headers_credentials,
                verify=verify
            )
            response.raise_for_status()
            embedding_data = response.json()
            embedding_vector = embedding_data['data'][0]['embedding']
            return embedding_vector
        except requests.exceptions.RequestException as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = None,
        dimensions: int = None
    ) -> List[List[float]]:
        """
        Get embedding vectors for multiple texts in a single request
        
        Args:
            texts: List of input texts
            model: Embedding model (defaults to constants.EMBEDDING_MODEL)
            dimensions: Output dimensions (defaults to constants.EMBEDDING_DIMENSIONS)
            
        Returns:
            List of embedding vectors
        """
        model = model or constants.EMBEDDING_MODEL
        dimensions = dimensions or constants.EMBEDDING_DIMENSIONS
        
        # Clean texts
        cleaned_texts = [(text or "").replace("\n", " ") for text in texts]
        
        payload = {
            'model': model,
            'input': cleaned_texts,
            'dimensions': dimensions,
            'encoding_format': 'float',
            'user': 'governance-agent'
        }
        
        try:
            verify = self.cert if self.cert else False
            response = requests.post(
                self.embedding_url,
                json=payload,
                headers=self.headers_credentials,
                verify=verify
            )
            response.raise_for_status()
            embedding_data = response.json()
            embeddings = [item['embedding'] for item in embedding_data['data']]
            return embeddings
        except requests.exceptions.RequestException as e:
            raise Exception(f"Batch embedding generation failed: {str(e)}")


# Global instance
cortex = CortexConnection()