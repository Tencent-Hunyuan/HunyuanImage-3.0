# -*- coding: utf-8 -*-
"""
MiniMax Client Module

This module provides a client interface for interacting with MiniMax Cloud API
for prompt enhancement (recaptioning) in HunyuanImage-3.0.

MiniMax offers an OpenAI-compatible API at https://api.minimax.io/v1 with
models like MiniMax-M2.7 and MiniMax-M2.5 that support large context windows.
"""
import json
import os
import time
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class MiniMaxClient(object):
    """
    Client for interacting with MiniMax Cloud API for prompt recaptioning.

    MiniMax provides an OpenAI-compatible API that can be used as an alternative
    to DeepSeek for prompt enhancement in image generation workflows.

    The client supports MiniMax-M2.7 (latest, 1M context) and MiniMax-M2.5
    (204K context) models.
    """

    BASE_URL = "https://api.minimax.io/v1"
    DEFAULT_MODEL = "MiniMax-M2.7"

    def __init__(self, api_key=None, model=None):
        """
        Initialize the MiniMax client.

        Args:
            api_key (str, optional): MiniMax API key. If not provided, reads
                from MINIMAX_API_KEY environment variable.
            model (str, optional): Model to use. Defaults to MiniMax-M2.7.
                Options: MiniMax-M2.7, MiniMax-M2.7-highspeed, MiniMax-M2.5,
                MiniMax-M2.5-highspeed.

        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for MiniMax provider. "
                "Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MiniMax API key is required. Set the MINIMAX_API_KEY "
                "environment variable or pass api_key to the constructor."
            )

        self.model = model or self.DEFAULT_MODEL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    def run_single_recaption(self, system_prompt, input_prompt):
        """
        Run a single prompt recaptioning request.

        This method sends a prompt to MiniMax API for enhancement/recaptioning,
        matching the same interface as DeepSeekClient.run_single_recaption().

        Args:
            system_prompt (str): System prompt that defines the task and behavior.
            input_prompt (str): User input prompt to be recaptioned/enhanced.

        Returns:
            tuple: A tuple containing:
                - content (str): The recaptioned/enhanced prompt.
                - reason (str): The reasoning content (empty string if the model
                    does not return reasoning).

        Note:
            The method includes retry logic to handle transient API errors.
            It will retry with a 1-second delay if an exception occurs.
        """
        print("Start to run recaption (MiniMax): ")

        # MiniMax temperature must be in (0.0, 1.0]
        temperature = 0.7

        # Retry loop to handle transient API errors
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_prompt},
                    ],
                    temperature=temperature,
                    stream=False,
                )
                break
            except Exception as e:
                logger.error(f"MiniMax API error: {e}")
                time.sleep(1)

        # Extract the enhanced prompt content
        choice = response.choices[0]
        content = choice.message.content or ""

        # Strip thinking tags if present (MiniMax-M2.7 may include them)
        content = _strip_think_tags(content)

        # MiniMax does not return separate reasoning content via the
        # standard OpenAI-compatible API, so we return an empty string
        reason = ""

        # Print debug information
        print("Initial prompt: ", input_prompt)
        print("Recaption prompt: ", content)

        return content, reason


def _strip_think_tags(text):
    """
    Remove <think>...</think> tags from model output.

    Some MiniMax models may include reasoning in <think> tags. This function
    strips those tags and returns only the final content.

    Args:
        text (str): The model output text.

    Returns:
        str: Text with think tags removed.
    """
    import re
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned.strip()
