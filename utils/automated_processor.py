import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from .data_models import ProcessingConfig
from .iterative_generator import IterativeGenerator
from llm.prompt_processor import OpenRouterPromptProcessor

class AutomatedPromptProcessor:
    """Automated prompt processor for predefined models"""
    
    
    def __init__(self, TARGET_MODELS):
        # Import here to avoid circular imports
        from utils.docker_service import DockerService
        self.docker_service = DockerService()
        self.processor = OpenRouterPromptProcessor(docker_service=self.docker_service)
        self.iterative_generator = IterativeGenerator(self.processor)
        self.TARGET_MODELS = TARGET_MODELS
    
    def run(self) -> None:
        """Main automated processing entry point"""
        print("🤖 Automated OpenRouter Prompt Processor")
        print("=" * 60)
        print(f"📋 Processing {len(self.TARGET_MODELS)} predefined models with 5 iterations each")
        print("=" * 60)
        
        try:
            # Get available models
            print("📡 Fetching available models...")
            available_models = self.processor.get_available_models()
            if not available_models:
                print("❌ No models available. Check your API key and internet connection.")
                return
            
            # Filter to only target models that are available
            available_model_ids = {model.get("id", "") for model in available_models}
            selected_models = []
            
            for target_model in self.TARGET_MODELS:
                if target_model in available_model_ids:
                    for model in available_models:
                        if model.get("id") == target_model:
                            selected_models.append(model)
                            print(f"✅ Found target model: {target_model}")
                            break
                else:
                    print(f"⚠️ Target model not available: {target_model}")
            
            if not selected_models:
                print("❌ None of the target models are available.")
                return
            
            print(f"\n🎯 Processing {len(selected_models)} available target models")
            
            # Read prompt
            print("\n📖 Reading prompt from generate.txt...")
            try:
                prompt = self.processor.read_prompt_from_file("prompt/generate.txt")
                print(f"✅ Prompt loaded ({len(prompt)} characters)")
            except Exception as e:
                print(f"❌ Error reading prompt: {e}")
                return
            
            # Create processing configuration
            config = ProcessingConfig(
                temperature=1.0,
                max_tokens=30000,  # Will be dynamically calculated
                k_iterations=5,
                prompt_file="prompt/generate.txt"
            )
            
            # Fixed bug: Better parallel processing with error handling
            print(f"\n🚀 Starting parallel processing of {len(selected_models)} models...")
            
            completed_models = []
            failed_models = []
            
            with ThreadPoolExecutor(max_workers=min(len(selected_models), 3)) as executor:  # Limit to 3 parallel
                futures = {}
                for model in selected_models:
                    future = executor.submit(self._process_model_safely, model, prompt, config)
                    futures[future] = model['id']
                
                for future in as_completed(futures):
                    model_id = futures[future]
                    try:
                        result = future.result()
                        if result:
                            print(f"✅ Completed processing for {model_id}")
                            completed_models.append(model_id)
                        else:
                            print(f"⚠️ Processing failed for {model_id}")
                            failed_models.append(model_id)
                    except Exception as e:
                        print(f"❌ Error in {model_id}: {e}")
                        failed_models.append(model_id)
            
            # Summary
            print(f"\n🎉 Automated processing complete!")
            print(f"✅ Successfully processed: {len(completed_models)}/{len(selected_models)} models")
            if failed_models:
                print(f"❌ Failed models: {', '.join(failed_models)}")
            print("📁 Check the 'bot/' directory for generated results")
            
        except KeyboardInterrupt:
            print("\n\n👋 Operation cancelled by user.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def _process_model_safely(self, selected_model: Dict[str, Any], prompt: str, config: ProcessingConfig) -> bool:
        """Process a single model in a thread-safe manner with dedicated instances"""
        # Create dedicated instances for thread safety
        from utils.docker_service import DockerService
        docker_service = DockerService()
        processor = OpenRouterPromptProcessor(docker_service=docker_service)
        iterative_generator = IterativeGenerator(processor)
        
        model_id = selected_model['id']
        print(f"\n🚀 Starting processing for model: {model_id}")
        
        try:
            best_result, best_bot_dir = iterative_generator.run_iterations(
                selected_model, prompt, config
            )
            
            if best_result and best_bot_dir:
                print(f"✅ Successfully completed iterative generation for {model_id}")
                return True
            else:
                print(f"⚠️ No successful iterations for {model_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error in processing {model_id}: {e}")
            return False
