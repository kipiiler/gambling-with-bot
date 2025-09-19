#!/usr/bin/env python3
"""
Simple and elegant main entry point for the OpenRouter Prompt Processor
"""

import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.automated_processor import AutomatedPromptProcessor

# TARGET_MODELS = [
#     "anthropic/claude-opus-4.1",
#     "qwen/qwen3-235b-a22b-2507",
#     "x-ai/grok-4",
#     "google/gemini-2.5-pro",
#     "anthropic/claude-sonnet-4",
#     "z-ai/glm-4.5",
#     "qwen/qwen3-coder",
#     "google/gemini-2.5-flash",
#     "moonshotai/kimi-k2",
#     "deepseek/deepseek-r1-0528",
#     "openai/gpt-5",
#     "openai/o3-pro"
# ]

TARGET_MODELS = [
    "deepseek/deepseek-r1-0528"
]

def main():
    """Main function to run automated processing of all target models"""
    print("🤖 OpenRouter Prompt Processor")
    print("=" * 50)
    print(f"🎯 Target Models: {len(TARGET_MODELS)}")
    print(f"🔄 Iterations per Model: 5")
    print(f"📊 Total Runs: {len(TARGET_MODELS) * 5}")
    print("=" * 50)
    
    # Initialize bot_directories.txt file
    with open("bot_directories.txt", 'w', encoding='utf-8') as f:
        f.write("# Model ID to Directory Path Mapping\n")
        f.write("# Generated automatically by main.py\n")
        f.write("# Format: model_id : dir_path\n\n")
    print("📝 Initialized bot_directories.txt")
    
    try:
        # Initialize and run the automated processor
        processor = AutomatedPromptProcessor(TARGET_MODELS)
        processor.run()
        
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
