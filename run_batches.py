#!/usr/bin/env python3
"""
Batch runner for OpenRouter Prompt Processor
Runs main.py multiple times and uploads generated bot files to API
"""

import sys
import os
import subprocess
import re
from typing import List

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.send_api import APISender

# Global configuration
BATCHES = 1
EXPECTED_DIRECTORIES_PER_BATCH = 1

def parse_created_directories(output_text: str) -> List[str]:
    """
    Parse the output from main.py to extract created directory paths.
    
    Args:
        output_text: The stdout/stderr output from running main.py
        
    Returns:
        List of directory paths that were created
    """
    pattern = r"📁 Created new directory: (.+)"
    directories = re.findall(pattern, output_text)
    return directories

def verify_required_files(directories: List[str]) -> bool:
    """
    Verify that each directory contains the required player.py and requirements.txt files.
    
    Args:
        directories: List of directory paths to check
        
    Returns:
        True if all directories have required files, False otherwise
    """
    required_files = ['player.py', 'requirements.txt']
    
    for directory in directories:
        for required_file in required_files:
            file_path = os.path.join(directory, required_file)
            if not os.path.isfile(file_path):
                print(f"❌ Missing required file '{required_file}' in directory '{directory}'")
                return False
        print(f"✅ Verified required files in: {directory}")
    
    return True

def run_single_batch(batch_num: int) -> List[str]:
    """
    Run a single batch of main.py and return the created directories.
    
    Args:
        batch_num: The batch number (for logging)
        
    Returns:
        List of created directory paths
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING BATCH {batch_num}/{BATCHES}")
    print(f"{'='*60}")
    
    # Run main.py and capture output
    try:
        process = subprocess.Popen(
            ["python", "main.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            universal_newlines=True
        )
        
        output_lines = []
        
        # Stream output in real-time while capturing it
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output, end='')  # Display in real-time
                output_lines.append(output)
        
        return_code = process.wait()
        
        if return_code != 0:
            print(f"❌ main.py exited with return code {return_code}")
            return []
        
        # Parse the captured output
        full_output = ''.join(output_lines)
        created_directories = parse_created_directories(full_output)
        
        print(f"\n📊 BATCH {batch_num} RESULTS:")
        print(f"   Expected directories: {EXPECTED_DIRECTORIES_PER_BATCH}")
        print(f"   Created directories: {len(created_directories)}")
        
        if len(created_directories) != EXPECTED_DIRECTORIES_PER_BATCH:
            print(f"⚠️  Warning: Expected {EXPECTED_DIRECTORIES_PER_BATCH} directories but found {len(created_directories)}")
        
        return created_directories
        
    except Exception as e:
        print(f"❌ Error running batch {batch_num}: {e}")
        return []

def main():
    """
    Main function to run all batches and upload results
    """
    print(f"🤖 Batch OpenRouter Prompt Processor")
    print(f"{'='*50}")
    print(f"🎯 Total Batches: {BATCHES}")
    print(f"📁 Expected Directories per Batch: {EXPECTED_DIRECTORIES_PER_BATCH}")
    print(f"📊 Total Expected Directories: {BATCHES * EXPECTED_DIRECTORIES_PER_BATCH}")
    print(f"{'='*50}")
    
    all_created_directories = []
    
    try:
        # Run all batches
        for batch_num in range(1, BATCHES + 1):
            batch_directories = run_single_batch(batch_num)
            
            if not batch_directories:
                print(f"❌ Batch {batch_num} failed to create any directories. Stopping.")
                sys.exit(1)
            
            all_created_directories.extend(batch_directories)
            print(f"✅ Batch {batch_num} completed successfully")
        
        print(f"\n{'='*60}")
        print(f"🎉 ALL BATCHES COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Total directories created: {len(all_created_directories)}")
        
        # Verify all directories have required files
        print(f"\n🔍 Verifying required files in all directories...")
        if not verify_required_files(all_created_directories):
            print(f"❌ Some directories are missing required files. Exiting program.")
            sys.exit(1)
        
        print(f"✅ All directories have required files!")
        
        # Upload files to API
        print(f"\n📤 Starting API uploads...")
        api_sender = APISender("https://api-huskyholdem.atcuw.org/submission/upload")
        
        success = api_sender.upload_all_directories(all_created_directories)
        
        if success:
            print(f"\n🎉 SUCCESS: All {len(all_created_directories)} bot directories uploaded successfully!")
        else:
            print(f"\n❌ FAILED: API upload process failed. Check logs above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
