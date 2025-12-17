from agents.models import ChatCompletionModel, ChatCompletionOutput, ChatCompletionChunk
from typing import List, Dict, Optional, Any, AsyncIterator
import json


class CortexChatCompletionsModel(ChatCompletionModel):
    """Custom model implementation for Cortex API that works with OpenAI Agents SDK"""
    
    def __init__(self, model: str, cortex_client):
        self.model = model
        self.client = cortex_client
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> ChatCompletionOutput:
        """Main completion method called by Agents SDK"""
        
        # Extract parameters
        temperature = kwargs.get('temperature', 0)
        max_tokens = kwargs.get('max_tokens', 1000)
        response_format = kwargs.get('response_format', None)
        
        # Convert messages to plain dicts
        clean_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                clean_messages.append({
                    'role': str(msg.get('role', 'user')),
                    'content': str(msg.get('content', ''))
                })
            else:
                clean_messages.append({
                    'role': 'user',
                    'content': str(msg)
                })
        
        # Call the Cortex client
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=clean_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        # Extract content
        content = response.choices[0].message.content if response.choices else ""
        
        # Return in format expected by Agents SDK
        return ChatCompletionOutput(
            content=content,
            stop_reason=response.choices[0].finish_reason if response.choices else "stop",
            model=self.model
        )
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Streaming not supported"""
        raise NotImplementedError("Streaming not supported for Cortex")