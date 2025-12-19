"""
API handler for Cortex API requests
"""
import requests
import json
from typing import Dict, Any, Optional


class APIHandler:
    """Handler for making API requests to Cortex"""
    
    def call_api_get(self, url: str, headers: Dict[str, str], cert: Optional[str] = None) -> requests.Response:
        """
        Make a GET request to the API
        
        Args:
            url: The API endpoint URL
            headers: Request headers
            cert: Path to CA certificate, or None to disable verification
            
        Returns:
            Response object
            
        Raises:
            requests.HTTPError: If the request fails
        """
        try:
            verify = cert if cert else False
            response = requests.get(url, headers=headers, verify=verify)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"GET request failed: {str(e)}")
    
    def call_api_post(
        self, 
        url: str, 
        headers: Dict[str, str], 
        payload: Dict[str, Any], 
        cert: Optional[str] = None
    ) -> requests.Response:
        """
        Make a POST request to the API
        
        Args:
            url: The API endpoint URL
            headers: Request headers
            payload: Request payload (will be JSON serialized)
            cert: Path to CA certificate, or None to disable verification
            
        Returns:
            Response object
            
        Raises:
            requests.HTTPError: If the request fails
        """
        try:
            verify = cert if cert else False
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload), 
                verify=verify
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"POST request failed: {str(e)}")
