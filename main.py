import sys

# Try to import Docker service
try:
    from docker_service import DockerService
    docker_service = DockerService()
    print("✅ Docker service initialized successfully")
except ImportError as e:
    print(f"⚠️  Docker service not available: {e}")
    print("   Game testing will be skipped")
    docker_service = None
except Exception as e:
    print(f"⚠️  Failed to initialize Docker service: {e}")
    print("   Game testing will be skipped")
    docker_service = None

from llm.prompt_processor import OpenRouterPromptProcessor


def main():
    """Main function to run the prompt processor"""
    print("🤖 OpenRouter Prompt Processor")
    print("=" * 50)
    
    try:
        # Initialize processor with Docker service
        processor = OpenRouterPromptProcessor(docker_service=docker_service)
        
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