# -*- coding: utf-8 -*-
"""
DeepSeek Client Module

This module provides a client interface for interacting with DeepSeek API
through Tencent Cloud's LKEAP service. It supports prompt recaptioning
with reasoning capabilities.
"""
import json
import time
import ast
from loguru import logger
from tencentcloud.common.common_client import CommonClient
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile


class NonStreamResponse(object):
    """
    Response handler for non-streaming API calls.
    
    This class is used to deserialize and store API responses in JSON format.
    """
    def __init__(self):
        """Initialize the response handler with an empty response string."""
        self.response = ""

    def _deserialize(self, obj):
        """
        Deserialize the response object to JSON string.
        
        Args:
            obj: The response object to be serialized
        """
        self.response = json.dumps(obj)


class DeepSeekClient(object):
    """
    Client for interacting with DeepSeek API through Tencent Cloud LKEAP service.
    
    This client provides functionality for prompt recaptioning with reasoning capabilities,
    enabling intelligent prompt enhancement for image generation tasks.
    """
    def __init__(self, key_id, key_secret):
        """
        Initialize the DeepSeek client with authentication credentials.
        
        Args:
            key_id (str): Tencent Cloud API key ID for authentication
            key_secret (str): Tencent Cloud API key secret for authentication
        """
        # Initialize credentials
        cred = credential.Credential(key_id, key_secret)
        
        # Configure HTTP profile with endpoint and timeout
        httpProfile = HttpProfile()
        httpProfile.endpoint = "lkeap.tencentcloudapi.com"
        # Set longer timeout for streaming interface compatibility
        httpProfile.reqTimeout = 40000  # The streaming interface may take a longer time.
        
        # Configure client profile
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        
        # Initialize the common client for LKEAP service
        self.common_client = CommonClient("lkeap", "2024-05-22", cred, "ap-guangzhou", profile=clientProfile)

    def run_single_recaption(self, system_prompt, input_prompt):
        """
        Run a single prompt recaptioning request with reasoning.
        
        This method sends a prompt to DeepSeek API for enhancement/recaptioning.
        It uses the thinking/reasoning capability to generate an improved prompt
        along with the reasoning process.
        
        Args:
            system_prompt (str): System prompt that defines the task and behavior
            input_prompt (str): User input prompt to be recaptioned/enhanced
        
        Returns:
            tuple: A tuple containing:
                - content (str): The recaptioned/enhanced prompt
                - reason (str): The reasoning content explaining the enhancement
        
        Note:
            The method includes retry logic to handle transient API errors.
            It will retry with a 1-second delay if an exception occurs.
        """
        # Prepare the API request payload
        post_dict = {
            "Model": "deepseek-v3.1",  # DeepSeek model version
            "Messages": [
                {
                    "Role": "system",
                    "Content": system_prompt
                },
                {
                    "Role": "user",
                    "Content": input_prompt
                }
            ],
            "Stream": False,  # Non-streaming response
            "Thinking": {"Type": "enabled"},  # Enable reasoning/thinking capability
        }
        
        print('Start to run recaption: ')
        
        # Retry loop to handle transient API errors
        while True:
            try:
                resp = self.common_client._call_and_deserialize("ChatCompletions", post_dict, NonStreamResponse)
                break
            except Exception as e:
                logger.error(e)
                time.sleep(1)  # Wait 1 second before retry
        
        # Make the actual API call (duplicate call for final response)
        resp = self.common_client._call_and_deserialize("ChatCompletions", post_dict, NonStreamResponse)
        response = resp.response
        
        # Parse the JSON response string to Python dict
        response = ast.literal_eval(response)
        
        # Extract the enhanced prompt content
        content = response["Choices"][0]["Message"]["Content"]
        # Extract the reasoning content
        reason = response["Choices"][0]["Message"]["ReasoningContent"]
        
        # Print debug information
        print('Initial prompt: ', input_prompt)
        print('Recaption prompt: ', content)

        return content, reason


if __name__ == "__main__":
    # This module is typically imported and used as a library
    # Main execution logic would be implemented in the calling script
    pass