#!/usr/bin/env python3

"""
Kimi Prompt Processor (Moonshot AI) - Fully Compatible with main.py

This script provides the exact same functionality as main.py but uses the Kimi API (Moonshot AI).
It supports:
- Interactive model selection
- Iterative generation with feedback
- Docker integration for game testing
- All the same parameters as main.py

Usage:
python kimi_prompt.py  # Interactive mode (same as main.py)
"""

import sys
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import re

# Import everything from main.py to reuse the functionality
from docker_service import DockerService
from config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_NUM_ITERATIONS

# Import all classes from main.py
from main import (
    ProcessingConfig,
    IterationResult,
    FeedbackAnalyzer,
    PromptBuilder,
    IterativeGenerator
)

# Load environment variables
load_dotenv()

class KimiPromptProcessor:
    """Kimi API processor that mimics OpenRouterPromptProcessor interface"""

    def __init__(self, docker_service=None):
        """Initialize the Kimi processor with same interface as OpenRouterPromptProcessor"""
        self.api_key = os.getenv("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY not found in .env file")
        
        self.base_url = "https://api.moonshot.ai/v1"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # Store Docker service if provided (same as OpenRouterPromptProcessor)
        self.docker_service = docker_service

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models using OpenAI SDK (compatible with OpenRouterPromptProcessor)"""
        try:
            response = self.client.models.list()
            # Convert to match OpenRouter format
            models = []
            for model in response.data:
                models.append({
                    "id": model.id,
                    "name": model.id,
                    "provider": {"id": "moonshot"},
                    "context_length": getattr(model, 'context_length', 128000)  # Default Kimi context length
                })
            return models
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def display_models(self, models: List[Dict[str, Any]]) -> None:
        """Display available models in a formatted list (compatible with OpenRouterPromptProcessor)"""
        print("\n=== Available Kimi Models ===")
        print(f"{'#':<3} {'Model ID':<40} {'Provider':<15} {'Context':<10}")
        print("-" * 70)
        
        for i, model in enumerate(models, 1):
            model_id = model.get("id", "Unknown")
            provider = model.get("provider", {}).get("id", "moonshot")
            context_length = model.get("context_length", "128K")
            
            print(f"{i:<3} {model_id:<40} {provider:<15} {context_length:<10}")

    def select_model(self, models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Allow user to select a model interactively (compatible with OpenRouterPromptProcessor)"""
        if not models:
            print("No models available.")
            return None
        
        while True:
            try:
                choice = input(f"\nSelect a model (1-{len(models)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]
                    print(f"✅ Selected: {selected_model['id']}")
                    return selected_model
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return None

    def process_prompt(self, model_id: str, prompt: str, temperature: float = 0.6, max_tokens: int = 9000) -> Dict[str, Any]:
        """Process prompt with specified model (compatible with OpenRouterPromptProcessor)"""
        messages = [
            {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        return response.model_dump()

    def read_prompt_from_file(self, file_path: str = "prompt/generate.txt") -> str:
        """Read prompt from file (compatible with OpenRouterPromptProcessor)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def extract_code_blocks(self, response_content: str) -> Tuple[str, str]:
        """Extract Python and text code blocks from the response (compatible with OpenRouterPromptProcessor)"""
        python_code = ""
        text_content = ""
        
        # Look for Python code blocks
        python_pattern = r'```python\n(.*?)\n```'
        python_matches = re.findall(python_pattern, response_content, re.DOTALL)
        if python_matches:
            python_code = python_matches[0].strip()
        
        # Look for requirements/text blocks
        text_pattern = r'```(?:txt|text|requirements)\n(.*?)\n```'
        text_matches = re.findall(text_pattern, response_content, re.DOTALL)
        if text_matches:
            text_content = text_matches[0].strip()
        
        # Fallback: if no specific language blocks found, get first two code blocks
        if not python_code or not text_content:
            all_blocks_pattern = r'```(?:(\w+)\n)?(.*?)\n```'
            all_blocks = re.findall(all_blocks_pattern, response_content, re.DOTALL)
            
            for i, (lang, content) in enumerate(all_blocks):
                if not python_code and (lang.lower() == 'python' or i == 0):
                    python_code = content.strip()
                elif not text_content and (lang.lower() in ['txt', 'text', 'requirements'] or (lang != 'python' and i == 1)):
                    text_content = content.strip()
        
        return python_code, text_content

    def save_code_blocks_to_bot_directory(self, python_code: str, text_content: str, model_id: str) -> str:
        """Save extracted code blocks to bot directory (compatible with OpenRouterPromptProcessor)"""
        try:
            if not os.path.exists("bot"):
                os.makedirs("bot")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model_id.replace("/", "_").replace("-", "_")
            dir_name = f"{model_name}_{timestamp}"
            dir_path = os.path.join("bot", dir_name)
            os.makedirs(dir_path, exist_ok=True)
            
            if python_code:
                player_path = os.path.join(dir_path, "player.py")
                with open(player_path, 'w', encoding='utf-8') as f:
                    f.write(python_code)
                print(f"✅ Python code saved to: {player_path}")
            
            if text_content:
                requirements_path = os.path.join(dir_path, "requirements.txt")
                with open(requirements_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                print(f"✅ Text content saved to: {requirements_path}")
            
            return dir_path
        except Exception as e:
            print(f"Error saving code blocks: {e}")
            return ""

    def save_result_to_log(self, result: Dict[str, Any], model_id: str, prompt: str, 
                          output_file: str = "output.log", save_code_blocks: bool = True) -> None:
        """Save result to log file (compatible with OpenRouterPromptProcessor)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_content = ""
        if "choices" in result and len(result["choices"]) > 0:
            response_content = result["choices"][0]["message"]["content"]
        
        log_entry = f"""
{'='*80}
TIMESTAMP: {timestamp}
MODEL: {model_id}
PROMPT LENGTH: {len(prompt)} characters
RESPONSE LENGTH: {len(response_content)} characters
USAGE: {result.get('usage', {})}
PROMPT:
{prompt}
RESPONSE:
{response_content}
RAW API RESPONSE:
{result}
{'='*80}
"""
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(f"✅ Result saved to {output_file}")
        print(f"📝 Response length: {len(response_content)} characters")
        
        if save_code_blocks and response_content:
            python_code, text_content = self.extract_code_blocks(response_content)
            if python_code or text_content:
                bot_dir = self.save_code_blocks_to_bot_directory(python_code, text_content, model_id)
                if bot_dir:
                    print(f"📁 Code blocks saved to: {bot_dir}")
            else:
                print("⚠️ No code blocks found in response")
        
        if response_content:
            print(f"\n📋 Preview (first 300 chars):")
            print(f"{response_content[:300]}...")

    def test_generated_code(self, bot_dir: str, client_name: str, iteration: int, port: Optional[int] = None) -> Tuple[bool, str]:
        """Test generated code using Docker service (compatible with OpenRouterPromptProcessor)"""
        if not self.docker_service:
            return False, "Docker service not available"
        
        try:
            # The OpenRouterPromptProcessor uses client_name as model_id in test_generated_code
            return self.docker_service.test_generated_code(bot_dir, client_name, iteration, port=port)
        except AttributeError:
            # If docker_service doesn't have test_generated_code method, call the actual method from processor
            # Import the method from the OpenRouterPromptProcessor temporarily
            from llm.prompt_processor import OpenRouterPromptProcessor
            temp_processor = OpenRouterPromptProcessor(docker_service=self.docker_service)
            return temp_processor.test_generated_code(bot_dir, client_name, iteration, port=port)
        except Exception as e:
            return False, str(e)


# Create a custom IterativeGenerator for Kimi
class KimiIterativeGenerator(IterativeGenerator):
    """Kimi-specific iterative generator that uses KimiPromptProcessor"""
    
    def __init__(self, processor: KimiPromptProcessor):
        self.processor = processor
        self.feedback_analyzer = FeedbackAnalyzer()
        self.prompt_builder = PromptBuilder()


class KimiPromptProcessorApp:
    """Main application class for the Kimi prompt processor - identical to main.py but for Kimi"""
    
    def __init__(self):
        self.docker_service = DockerService()
        self.processor = KimiPromptProcessor(docker_service=self.docker_service)
        self.iterative_generator = KimiIterativeGenerator(self.processor)
    
    def select_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select multiple models from the list"""
        print("\nEnter model numbers (comma-separated for multiple, e.g., 1,3,5) or 0 to exit:")
        selection = input().strip()
        
        if selection == '0':
            return []
        
        try:
            indices = [int(s.strip()) for s in selection.split(',') if s.strip()]
        except ValueError:
            print("Invalid input. Exiting.")
            return []
        
        selected = []
        for idx in indices:
            if 1 <= idx <= len(models):
                selected.append(models[idx-1])
            else:
                print(f"Invalid model number: {idx}")
        
        if not selected:
            print("No valid models selected. Exiting.")
            return []
        
        return selected
    
    def _process_model(self, selected_model: Dict[str, Any], prompt: str, config: ProcessingConfig) -> None:
        """Process a single model in a thread-safe manner with dedicated instances"""
        docker_service = DockerService()
        processor = KimiPromptProcessor(docker_service=docker_service)
        iterative_generator = KimiIterativeGenerator(processor)
        
        model_id = selected_model['id']
        print(f"\n🚀 Starting processing for model: {model_id}")
        
        try:
            if config.k_iterations == 1:
                print(f"Running single generation for {model_id}...")
                result = processor.process_prompt(
                    model_id=selected_model['id'],
                    prompt=prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
                processor.save_result_to_log(result, selected_model['id'], prompt)
                print(f"✅ Completed single generation for {model_id}")
            else:
                print(f"Running iterative generation for {model_id} with {config.k_iterations} iterations...")
                iterative_generator.run_iterations(
                    selected_model, prompt, config
                )
                print(f"✅ Completed iterative generation for {model_id}")
        except Exception as e:
            print(f"❌ Error in processing {model_id}: {e}")
            raise
    
    def run(self) -> None:
        """Main application entry point"""
        print("🤖 Kimi Prompt Processor (Moonshot AI)")
        print("=" * 50)
        
        try:
            # Get available models
            print("📡 Fetching available models...")
            models = self.processor.get_available_models()
            
            if not models:
                print("❌ No models available. Check your API key and internet connection.")
                return
            
            # Display models and get selection
            self.processor.display_models(models)
            selected_models = self.select_models(models)
            if not selected_models:
                print("👋 Goodbye!")
                return
            
            # Read prompt from file
            print("\n📖 Reading prompt from generate.txt...")
            try:
                prompt = self.processor.read_prompt_from_file("prompt/generate.txt")
                print(f"✅ Prompt loaded ({len(prompt)} characters)")
            except Exception as e:
                print(f"❌ Error reading prompt: {e}")
                return
            
            # Get processing parameters
            config = self._get_processing_config()
            
            if len(selected_models) == 1:
                selected_model = selected_models[0]
                # Choose between single or iterative generation
                if config.k_iterations == 1:
                    self._run_single_generation(selected_model, prompt, config)
                else:
                    self._run_iterative_generation(selected_model, prompt, config)
            else:
                # Parallel processing for multiple models
                with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
                    futures = {}
                    for model in selected_models:
                        future = executor.submit(self._process_model, model, prompt, config)
                        futures[future] = model['id']
                    
                    for future in as_completed(futures):
                        model_id = futures[future]
                        try:
                            future.result()
                            print(f"✅ Completed processing for {model_id}")
                        except Exception as e:
                            print(f"❌ Error in {model_id}: {e}")
            
            print("\n🎉 Processing complete!")
            
        except KeyboardInterrupt:
            print("\n\n👋 Operation cancelled by user.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    
    def _get_processing_config(self) -> ProcessingConfig:
        """Get processing parameters from user input"""
        print("\n⚙️  Processing Parameters:")
        try:
            temperature = float(input(f"Temperature (0.0-2.0, default {DEFAULT_TEMPERATURE}): ") or str(DEFAULT_TEMPERATURE))
            max_tokens = int(input(f"Max tokens (default {DEFAULT_MAX_TOKENS}): ") or str(DEFAULT_MAX_TOKENS))
            k_iterations = int(input(f"Number of iterations (default {DEFAULT_NUM_ITERATIONS}): ") or str(DEFAULT_NUM_ITERATIONS))
        except ValueError:
            print("Using default parameters...")
            temperature = DEFAULT_TEMPERATURE
            max_tokens = DEFAULT_MAX_TOKENS
            k_iterations = DEFAULT_NUM_ITERATIONS
        
        return ProcessingConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            k_iterations=k_iterations
        )
    
    def _run_single_generation(self, selected_model: Dict[str, Any], prompt: str, config: ProcessingConfig) -> None:
        """Run single generation (original behavior)"""
        print(f"\n🚀 Processing prompt with {selected_model['id']}...")
        print("⏳ This may take a moment...")
        
        result = self.processor.process_prompt(
            model_id=selected_model['id'],
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Save result
        self.processor.save_result_to_log(result, selected_model['id'], prompt)
        
        # Show preview
        if "choices" in result and len(result["choices"]) > 0:
            response_content = result["choices"][0]["message"]["content"]
            print(f"\n📋 Response Preview (first 200 chars):")
            print(f"{response_content[:200]}...")
    
    def _run_iterative_generation(self, selected_model: Dict[str, Any], prompt: str, config: ProcessingConfig) -> None:
        """Run iterative generation"""
        best_result, best_bot_dir = self.iterative_generator.run_iterations(
            selected_model, prompt, config
        )


def main():
    """Main function to run the Kimi prompt processor"""
    app = KimiPromptProcessorApp()
    app.run()


if __name__ == "__main__":
    main()