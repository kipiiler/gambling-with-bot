#!/usr/bin/env python3
"""
Simple and elegant main entry point for the Hermes Prompt Processor
"""

import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.hermes_processor import HermesAutomatedProcessor

# TARGET_MODELS for Hermes
TARGET_MODELS = [
    "hermes/hermes-4-405b",
]

def main():
    """Main function to run automated processing of Hermes models"""
    print("🤖 Hermes Prompt Processor")
    print("=" * 50)
    print(f"🎯 Target Models: {len(TARGET_MODELS)}")
    print(f"🔄 Iterations per Model: 5")
    print(f"📊 Total Runs: {len(TARGET_MODELS) * 5}")
    print("=" * 50)
    
    # Initialize bot_directories.txt file
    with open("bot_directories.txt", 'w', encoding='utf-8') as f:
        f.write("# Model ID to Directory Path Mapping\n")
        f.write("# Generated automatically by hermes.py\n")
        f.write("# Format: model_id : dir_path\n\n")
    print("📝 Initialized bot_directories.txt")
    
    try:
        # Initialize and run the automated processor
        processor = HermesAutomatedProcessor(TARGET_MODELS)
        processor.run()
        
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
