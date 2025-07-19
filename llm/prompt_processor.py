#!/usr/bin/env python3
"""
Prompt Processor Script for OpenRouter API

This script allows users to:
1. List available models from OpenRouter
2. Choose a model interactively
3. Process a prompt from generate.txt
4. Save results to output.log
5. Extract code blocks and save to bot directory
6. Test the generated code by running a game with Docker
"""

import os
import json
import sys
import re
import io
import tarfile
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Import Docker service
try:
    from docker import DockerService
except ImportError:
    print("Warning: Docker service not available. Game testing will be skipped.")
    DockerService = None

class OpenRouterPromptProcessor:
    """Process prompts through OpenRouter API with model selection"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the processor
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY in .env file")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gambling-with-bot",
            "X-Title": "Prompt Processor"
        }
        
        # Initialize Docker service if available
        self.docker_service = DockerService() if DockerService else None
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get models: {response.status_code} - {response.text}")
            
            models_data = response.json()
            return models_data.get("data", [])
            
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def display_models(self, models: List[Dict[str, Any]]) -> None:
        """Display available models in a formatted list"""
        print("\n=== Available Models ===")
        print(f"{'#':<3} {'Model ID':<40} {'Provider':<15} {'Context':<10}")
        print("-" * 70)
        
        for i, model in enumerate(models, 1):
            model_id = model.get("id", "Unknown")
            provider = model.get("provider", {}).get("id", "Unknown")
            context_length = model.get("context_length", "N/A")
            
            print(f"{i:<3} {model_id:<40} {provider:<15} {context_length:<10}")
    
    def select_model(self, models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Allow user to select a model interactively"""
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
                    print(f"\nSelected: {selected_model['id']}")
                    return selected_model
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
                    
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None
    
    def process_prompt(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Process a prompt through the selected model
        
        Args:
            model_id: The model to use
            prompt: The prompt text to process
            **kwargs: Additional parameters for the API call
            
        Returns:
            API response as dictionary
        """
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4000),
            "stream": False
        }
        
        # Add optional parameters
        for key in ["top_p", "frequency_penalty", "presence_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
            
            return response.json()
            
        except Exception as e:
            raise Exception(f"Error processing prompt: {e}")
    
    def read_prompt_from_file(self, file_path: str = "prompt/generate.txt") -> str:
        """Read prompt from the specified file"""
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
        
        Args:
            response_content: The response content from the LLM
            
        Returns:
            Tuple of (python_code, text_content)
        """
        python_code = ""
        text_content = ""
        
        # Extract Python code blocks
        python_pattern = r'```python\s*\n(.*?)\n```'
        python_matches = re.findall(python_pattern, response_content, re.DOTALL)
        if python_matches:
            python_code = python_matches[0].strip()
        
        # Extract text blocks (non-Python code blocks)
        text_pattern = r'```text\s*\n(.*?)\n```'
        text_matches = re.findall(text_pattern, response_content, re.DOTALL)
        if text_matches:
            text_content = text_matches[0].strip()
        
        # If no text block found, try to extract any non-Python code block
        if not text_content:
            # Find all code blocks
            all_blocks_pattern = r'```(\w+)\s*\n(.*?)\n```'
            all_blocks = re.findall(all_blocks_pattern, response_content, re.DOTALL)
            
            for lang, content in all_blocks:
                if lang.lower() != 'python':
                    text_content = content.strip()
                    break
        
        return python_code, text_content
    
    def create_tar_stream(self, player_code: str, requirements_content: str) -> io.BytesIO:
        """Create a tar stream with player.py and requirements.txt"""
        tar_stream = io.BytesIO()
        
        with tarfile.open(fileobj=tar_stream, mode='w:gz') as tar:
            # Add player.py
            if player_code:
                player_info = tarfile.TarInfo(name="player.py")
                player_info.size = len(player_code.encode('utf-8'))
                tar.addfile(player_info, io.BytesIO(player_code.encode('utf-8')))
            
            # Add requirements.txt
            if requirements_content:
                req_info = tarfile.TarInfo(name="requirements.txt")
                req_info.size = len(requirements_content.encode('utf-8'))
                tar.addfile(req_info, io.BytesIO(requirements_content.encode('utf-8')))
        
        tar_stream.seek(0)
        return tar_stream
    
    def test_generated_code(self, bot_dir: str, model_id: str) -> Tuple[bool, str]:
        """
        Test the generated code by running a game with Docker
        
        Args:
            bot_dir: Directory containing the generated code
            model_id: Model ID for logging
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.docker_service:
            return False, "Docker service not available"
        
        try:
            print(f"\n🧪 Testing generated code from {model_id}...")
            
            # Read the generated files
            player_path = os.path.join(bot_dir, "player.py")
            requirements_path = os.path.join(bot_dir, "requirements.txt")
            
            if not os.path.exists(player_path):
                return False, "player.py not found"
            
            with open(player_path, 'r', encoding='utf-8') as f:
                player_code = f.read()
            
            requirements_content = ""
            if os.path.exists(requirements_path):
                with open(requirements_path, 'r', encoding='utf-8') as f:
                    requirements_content = f.read()
            
            # Create tar stream with the files
            tar_stream = self.create_tar_stream(player_code, requirements_content)
            
            # Generate a unique test ID
            test_id = f"test_{model_id.replace('/', '_').replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start game server
            port = self.docker_service.generate_random_port()
            server_container_name = self.docker_service.get_game_server_container_name(port)
            
            print(f"🚀 Starting game server on port {port}...")
            if not self.docker_service.start_game_server(port, sim=True, num_bot=1):
                return False, "Failed to start game server"
            
            # Wait for server to be ready
            import time
            time.sleep(5)
            
            # Start default client
            print("🤖 Starting default client...")
            if not self.docker_service.start_client_container("localhost", port, "default", sim=True):
                return False, "Failed to start default client"
            
            # Start test client with our code
            print("🧪 Starting test client with generated code...")
            test_client_name = f"test_client_{test_id}"
            if not self.docker_service.start_client_container("localhost", port, test_client_name, sim=True):
                return False, "Failed to start test client"
            
            # Get test client container name
            formatted_test_id = self.docker_service.format_username(test_client_name)
            test_client_container_name = f"client_container_{port}_{formatted_test_id}"
            
            # Load our code into the test client
            print("📦 Loading generated code into test client...")
            if not self.docker_service.load_file_into_container(test_client_container_name, tar_stream):
                return False, "Failed to load code into test client"
            
            # Install requirements
            print("📦 Installing requirements...")
            install_result = self.docker_service.install_python_package(test_client_container_name)
            if install_result != "success":
                return False, f"Failed to install requirements: {install_result}"
            
            # Validate the code
            print("🔍 Validating generated code...")
            validation_result = self.docker_service.malform_file_client_check(test_client_container_name)
            if validation_result != "success":
                return False, f"Code validation failed: {validation_result}"
            
            # Wait for game to complete
            print("⏳ Waiting for game to complete...")
            max_wait_time = 300  # 5 minutes
            wait_time = 0
            while wait_time < max_wait_time:
                status = self.docker_service.check_game_container_status(port, sim=True)
                if status.value == "DONE":
                    break
                time.sleep(10)
                wait_time += 10
                print(f"⏳ Game status: {status.value} ({wait_time}s elapsed)")
            
            if wait_time >= max_wait_time:
                return False, "Game timed out"
            
            # Collect results
            print("📊 Collecting game results...")
            success, error_msg = self.collect_game_results(
                bot_dir, port, test_client_container_name, server_container_name
            )
            
            # Cleanup
            print("🧹 Cleaning up containers...")
            self.docker_service.stop_and_remove_container(test_client_container_name)
            self.docker_service.stop_and_remove_container(server_container_name)
            self.docker_service.release_port(port)
            
            return success, error_msg
            
        except Exception as e:
            return False, f"Game test failed with exception: {str(e)}"
    
    def collect_game_results(self, bot_dir: str, port: int, test_client_container: str, server_container: str) -> Tuple[bool, str]:
        """Collect game results and save to verified directory"""
        try:
            # Create verified directory
            verified_dir = os.path.join(bot_dir, "verified")
            os.makedirs(verified_dir, exist_ok=True)
            
            # Collect errors from test client
            error_log_path = os.path.join(verified_dir, "error.log")
            errors = []
            
            try:
                # Get container logs
                logs = self.docker_service.get_container_logs(test_client_container, tail=100)
                if "error" in logs.lower() or "exception" in logs.lower():
                    errors.append(f"Container logs contain errors:\n{logs}")
                
                # Check for any error files
                exit_code, output = self.docker_service._get_container(test_client_container).exec_run("find /app -name '*.log' -exec grep -l -i error {} \\;")
                if exit_code == 0 and output:
                    error_files = output.decode('utf-8').splitlines()
                    for error_file in error_files:
                        _, error_content = self.docker_service._get_container(test_client_container).exec_run(f"cat {error_file}")
                        errors.append(f"Error in {error_file}:\n{error_content.decode('utf-8')}")
                
            except Exception as e:
                errors.append(f"Error collecting logs: {str(e)}")
            
            # Try to get game log
            try:
                success, error_msg = self.docker_service.run_game_and_save_log(server_container, 2, verified_dir)
                if not success:
                    errors.append(f"Failed to get game log: {error_msg}")
            except Exception as e:
                errors.append(f"Exception getting game log: {str(e)}")
            
            # Save error log
            with open(error_log_path, 'w', encoding='utf-8') as f:
                if errors:
                    f.write(f"Game Test Errors for {datetime.now()}\n")
                    f.write("=" * 50 + "\n\n")
                    for error in errors:
                        f.write(f"{error}\n\n")
                else:
                    f.write("No errors detected during game test.\n")
            
            return len(errors) == 0, "Game test completed"
            
        except Exception as e:
            return False, f"Failed to collect results: {str(e)}"
    
    def save_code_blocks_to_bot_directory(self, python_code: str, text_content: str, model_id: str) -> str:
        """
        Save extracted code blocks to bot directory with specified structure
        
        Args:
            python_code: Python code to save as player.py
            text_content: Text content to save as requirements.txt
            model_id: Model ID for directory naming
            
        Returns:
            Path to the created directory
        """
        try:
            # Create bot directory if it doesn't exist
            if not os.path.exists("bot"):
                os.makedirs("bot")
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Clean model name for directory
            model_name = model_id.replace("/", "_").replace("-", "_")
            
            # Create directory name
            dir_name = f"{model_name}_{timestamp}"
            dir_path = os.path.join("bot", dir_name)
            
            # Create directory
            os.makedirs(dir_path, exist_ok=True)
            
            # Save player.py
            player_path = os.path.join(dir_path, "player.py")
            with open(player_path, 'w', encoding='utf-8') as f:
                f.write(python_code)
            print(f"✅ Python code saved to: {player_path}")
            
            # Save requirements.txt
            requirements_path = os.path.join(dir_path, "requirements.txt")
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"✅ Text content saved to: {requirements_path}")
            
            return dir_path
            
        except Exception as e:
            print(f"Error saving code blocks: {e}")
            return ""
    
    def save_result_to_log(self, result: Dict[str, Any], model_id: str, prompt: str, output_file: str = "output.log") -> None:
        """Save the result to output.log with timestamp and metadata"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract the response content
            response_content = ""
            if "choices" in result and len(result["choices"]) > 0:
                response_content = result["choices"][0]["message"]["content"]
            
            # Prepare log entry
            log_entry = f"""
{'='*80}
TIMESTAMP: {timestamp}
MODEL: {model_id}
PROMPT LENGTH: {len(prompt)} characters
RESPONSE LENGTH: {len(response_content)} characters
USAGE: {json.dumps(result.get('usage', {}), indent=2)}

PROMPT:
{prompt}

RESPONSE:
{response_content}

RAW API RESPONSE:
{json.dumps(result, indent=2)}
{'='*80}

"""
            
            # Write to log file
            with open(output_file, 'a', encoding='utf-8') as file:
                file.write(log_entry)
            
            print(f"\n✅ Result saved to {output_file}")
            print(f"📝 Response length: {len(response_content)} characters")
            print(f"💾 Usage: {result.get('usage', {})}")
            
            # Extract and save code blocks
            if response_content:
                python_code, text_content = self.extract_code_blocks(response_content)
                
                if python_code or text_content:
                    bot_dir = self.save_code_blocks_to_bot_directory(python_code, text_content, model_id)
                    if bot_dir:
                        print(f"📁 Code blocks saved to: {bot_dir}")
                        
                        # Test the generated code
                        if self.docker_service:
                            test_success, test_error = self.test_generated_code(bot_dir, model_id)
                            if test_success:
                                print("✅ Game test completed successfully!")
                            else:
                                print(f"⚠️  Game test failed: {test_error}")
                        else:
                            print("⚠️  Docker service not available - skipping game test")
                else:
                    print("⚠️  No code blocks found in response")
            
        except Exception as e:
            print(f"Error saving to log: {e}")


def main():
    """Main function to run the prompt processor"""
    print("🤖 OpenRouter Prompt Processor")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = OpenRouterPromptProcessor()
        
        # Get available models
        print("📡 Fetching available models...")
        models = processor.get_available_models()
        
        if not models:
            print("❌ No models available. Check your API key and internet connection.")
            return
        
        # Display models
        processor.display_models(models)
        
        # Select model
        selected_model = processor.select_model(models)
        if not selected_model:
            print("👋 Goodbye!")
            return
        
        # Read prompt from file
        print("\n📖 Reading prompt from generate.txt...")
        try:
            prompt = processor.read_prompt_from_file("prompt/generate.txt")
            print(f"✅ Prompt loaded ({len(prompt)} characters)")
        except Exception as e:
            print(f"❌ Error reading prompt: {e}")
            return
        
        # Get processing parameters
        print("\n⚙️  Processing Parameters:")
        try:
            temperature = float(input("Temperature (0.0-2.0, default 0.7): ") or "0.7")
            max_tokens = int(input("Max tokens (default 4000): ") or "4000")
        except ValueError:
            print("Using default parameters...")
            temperature = 0.7
            max_tokens = 4000
        
        # Process the prompt
        print(f"\n🚀 Processing prompt with {selected_model['id']}...")
        print("⏳ This may take a moment...")
        
        result = processor.process_prompt(
            model_id=selected_model['id'],
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Save result
        processor.save_result_to_log(result, selected_model['id'], prompt)
        
        # Show preview
        if "choices" in result and len(result["choices"]) > 0:
            response_content = result["choices"][0]["message"]["content"]
            print(f"\n📋 Response Preview (first 200 chars):")
            print(f"{response_content[:200]}...")
        
        print("\n🎉 Processing complete!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()