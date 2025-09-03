import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from utils.data_models import ProcessingConfig
from utils.iterative_generator import IterativeGenerator

class HermesAutomatedProcessor:
    """Automated prompt processor for Hermes models"""
    
    def __init__(self, TARGET_MODELS):
        from utils.docker_service import DockerService
        from llm.hermes_client import HermesProcessor
        
        self.docker_service = DockerService()
        self.TARGET_MODELS = TARGET_MODELS
        
        # Initialize Hermes processor
        self.hermes_processor = HermesProcessor(docker_service=self.docker_service)

    def run(self) -> None:
        """Main automated processing entry point"""
        print("🤖 Automated Hermes Prompt Processor")
        print("=" * 60)
        print(f"📋 Processing {len(self.TARGET_MODELS)} Hermes models")
        print("=" * 60)
        
        try:
            # Get available models
            print("📡 Getting available Hermes models...")
            available_models = self.hermes_processor.get_available_models()
            
            # Validate models
            available_model_ids = [model['id'] for model in available_models]
            print(f"✅ Available models: {available_model_ids}")
            
            invalid_models = [model for model in self.TARGET_MODELS if model not in available_model_ids]
            if invalid_models:
                print(f"⚠️ Invalid models (skipping): {invalid_models}")
            
            valid_models = [model for model in self.TARGET_MODELS if model in available_model_ids]
            
            if not valid_models:
                print("❌ No valid models to process")
                return
            
            print(f"🎯 Processing {len(valid_models)} valid models: {valid_models}")
            
            # Convert model IDs to model dictionaries
            selected_models = []
            for model_id in valid_models:
                model_dict = next((m for m in available_models if m['id'] == model_id), None)
                if model_dict:
                    selected_models.append(model_dict)
            
            # Read prompt
            print("\n📖 Reading prompt from generate.txt...")
            try:
                prompt = self.hermes_processor.read_prompt_from_file("prompt/generate.txt")
                print(f"✅ Prompt loaded ({len(prompt)} characters)")
            except Exception as e:
                print(f"❌ Error reading prompt: {e}")
                return
            
            # Create processing configuration
            config = ProcessingConfig(
                temperature=1.0,
                max_tokens=30000,
                k_iterations=5,
                prompt_file="prompt/generate.txt"
            )
            
            # Process each model
            completed_models = []
            failed_models = []
            
            for model in selected_models:
                model_id = model['id']
                print(f"\n{'='*60}")
                print(f"🚀 Processing Model: {model_id}")
                print(f"{'='*60}")
                
                try:
                    result = self._process_model_safely(model, prompt, config)
                    if result:
                        print(f"✅ Successfully completed processing for {model_id}")
                        completed_models.append(model_id)
                    else:
                        print(f"⚠️ Processing failed for {model_id}")
                        failed_models.append(model_id)
                        
                except Exception as e:
                    print(f"❌ Error processing {model_id}: {e}")
                    failed_models.append(model_id)
                    continue
            
            # Summary
            print(f"\n🎉 Hermes processing complete!")
            print(f"✅ Successfully processed: {len(completed_models)}/{len(selected_models)} models")
            if failed_models:
                print(f"❌ Failed models: {', '.join(failed_models)}")
            print("📁 Check the 'bot/' directory for generated results")
                    
        except Exception as e:
            print(f"❌ Fatal error in automated processing: {e}")
            raise
    
    def _process_model_safely(self, selected_model: Dict[str, Any], prompt: str, config: ProcessingConfig) -> bool:
        """Process a single model with the appropriate processor"""
        # Create dedicated instances for thread safety
        from utils.docker_service import DockerService
        from llm.hermes_client import HermesProcessor
        
        docker_service = DockerService()
        model_id = selected_model['id']
        
        # Create Hermes processor
        processor = HermesProcessor(docker_service=docker_service)
        print(f"\n🤖 Starting Hermes processing for: {model_id}")
        
        # Create iterative generator with the processor
        iterative_generator = IterativeGenerator(processor)
        
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
