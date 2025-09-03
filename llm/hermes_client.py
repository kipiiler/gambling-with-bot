#!/usr/bin/env python3
"""
Hermes-4-405B Client using Nous Research API
Uses the same pattern as OpenAI client but connects to Nous Research endpoint
"""

import os
import json
import requests
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class HermesProcessor:
    """Hermes-4-405B processor using Nous Research API"""
    
    def __init__(self, docker_service=None):
        self.base_url = "https://inference-api.nousresearch.com/v1"
        self.api_key = os.getenv("HERMES_API_KEY")
        if not self.api_key:
            raise ValueError("HERMES_API_KEY not found in environment variables")
        
        self.docker_service = docker_service
        
        # Initialize OpenAI client with Nous Research endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available Hermes models"""
        return [
            {
                "id": "hermes/hermes-4-405b",
                "name": "Hermes-4-405B",
                "description": "Hermes-4-405B via Nous Research API"
            }
        ]

    def process_prompt(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Process prompt using Hermes-4-405B
        Returns: API response dict compatible with other processors
        """
        # Extract config values from kwargs
        max_tokens = kwargs.get('max_tokens', 8192)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.8)
        presence_penalty = kwargs.get('presence_penalty', 1.5)
        
        print(f"🤖 Processing with Hermes-4-405B...")
        print(f"📊 Config: tokens={max_tokens}, temp={temperature}, top_p={top_p}")
        
        try:
            # Make API call using OpenAI client
            chat_response = self.client.chat.completions.create(
                model="Hermes-4-405B",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                extra_body={
                    "chat_template_kwargs": {"thinking": True},
                },
            )
            
            # Convert response to dict format compatible with other processors
            response_dict = {
                "choices": [
                    {
                        "message": {
                            "content": chat_response.choices[0].message.content
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": getattr(chat_response.usage, 'prompt_tokens', 0) if hasattr(chat_response, 'usage') else 0,
                    "completion_tokens": getattr(chat_response.usage, 'completion_tokens', 0) if hasattr(chat_response, 'usage') else 0,
                    "total_tokens": getattr(chat_response.usage, 'total_tokens', 0) if hasattr(chat_response, 'usage') else 0
                },
                "model": "Hermes-4-405B"
            }
            
            print(f"✅ Received response ({len(response_dict['choices'][0]['message']['content'])} chars)")
            
            return response_dict
            
        except Exception as e:
            print(f"❌ Hermes API Error: {e}")
            raise

    def read_prompt_from_file(self, file_path: str = "prompt/generate.txt") -> str:
        """Read prompt from file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error reading prompt file: {e}")

    def extract_code_blocks(self, response_content: str) -> Tuple[str, str]:
        """
        Extract Python and text code blocks from the response
        Reuses the same logic as other processors
        """
        from llm.prompt_processor import OpenRouterPromptProcessor
        temp_processor = OpenRouterPromptProcessor()
        return temp_processor.extract_code_blocks(response_content)

    def save_result_to_log(self, result: Dict[str, Any], model_id: str, prompt: str, output_file: str = "output.log", save_code_blocks: bool = True) -> None:
        """Save result to log - delegate to OpenRouter processor"""
        from llm.prompt_processor import OpenRouterPromptProcessor
        temp_processor = OpenRouterPromptProcessor()
        return temp_processor.save_result_to_log(result, model_id, prompt, output_file, save_code_blocks)

    # Delegate methods to reuse existing functionality
    def test_generated_code(self, bot_dir: str, model_id: str, iteration: int = 1, port: Optional[int] = None):
        """Delegate to OpenRouter processor's Docker testing logic"""
        if not self.docker_service:
            return False, "Docker service not available"
        
        from llm.prompt_processor import OpenRouterPromptProcessor
        temp_processor = OpenRouterPromptProcessor(docker_service=self.docker_service)
        return temp_processor.test_generated_code(bot_dir, model_id, iteration, port)

    def save_code_blocks_to_bot_directory(self, python_code: str, text_content: str, model_id: str, existing_dir: str = None) -> str:
        """Delegate to OpenRouter processor's directory creation logic"""
        from llm.prompt_processor import OpenRouterPromptProcessor
        temp_processor = OpenRouterPromptProcessor()
        return temp_processor.save_code_blocks_to_bot_directory(python_code, text_content, model_id, existing_dir)

    def create_tar_stream(self, player_code: str, requirements_content: str):
        """Delegate to OpenRouter processor"""
        from llm.prompt_processor import OpenRouterPromptProcessor
        temp_processor = OpenRouterPromptProcessor()
        return temp_processor.create_tar_stream(player_code, requirements_content)

    def cleanup_containers(self, test_client_container_name=None, server_container_name=None, port=None, release_port=True):
        """Delegate to OpenRouter processor"""
        from llm.prompt_processor import OpenRouterPromptProcessor
        temp_processor = OpenRouterPromptProcessor(docker_service=self.docker_service)
        return temp_processor.cleanup_containers(test_client_container_name, server_container_name, port, release_port)
