#!/usr/bin/env python3
"""
HuskyBench Submission Script
Automatically submits all bots from bot_directories.txt to HuskyBench API
"""

import json
import requests
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('huskybench_submissions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HuskyBenchSubmitter:
    """Handles submission of bots to HuskyBench API"""
    
    def __init__(self, base_url: str = "https://api.huskybench.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.auth_token = None
        
    def login(self, username: str, password: str) -> bool:
        """Login to HuskyBench API"""
        try:
            login_url = f"{self.base_url}/auth/login"
            login_data = {
                "username": username,
                "password": password
            }
            
            logger.info(f"Attempting login for user: {username}")
            response = self.session.post(login_url, json=login_data)
            
            if response.status_code == 200:
                result = response.json()
                # Get access_token from response based on API test
                self.auth_token = result.get('access_token')
                if self.auth_token:
                    # Set authorization header for future requests
                    self.session.headers.update({
                        'Authorization': f'Bearer {self.auth_token}'
                    })
                logger.info(f"Successfully logged in as {username}")
                return True
            else:
                logger.error(f"Login failed for {username}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Login error for {username}: {str(e)}")
            return False
    
    def upload_submission(self, player_py_path: str, requirements_txt_path: str) -> Optional[str]:
        """Upload player.py and requirements.txt files"""
        try:
            upload_url = f"{self.base_url}/submission/upload"
            
            # Prepare files for upload
            files = {}
            
            # Read and prepare player.py
            if os.path.exists(player_py_path):
                with open(player_py_path, 'rb') as f:
                    files['python_file'] = ('player.py', f.read(), 'text/plain')
            else:
                logger.error(f"player.py not found at {player_py_path}")
                return None
            
            # Read and prepare requirements.txt
            if os.path.exists(requirements_txt_path):
                with open(requirements_txt_path, 'rb') as f:
                    files['packages_file'] = ('requirements.txt', f.read(), 'text/plain')
            else:
                logger.error(f"requirements.txt not found at {requirements_txt_path}")
                return None
            
            logger.info(f"Uploading files: {player_py_path}, {requirements_txt_path}")
            response = self.session.post(upload_url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                submission_id = result.get('submission_id') or result.get('id')
                logger.info(f"Successfully uploaded submission. ID: {submission_id}")
                return submission_id
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return None
    
    def mark_final(self, submission_id: str) -> bool:
        """Mark a submission as final"""
        try:
            final_url = f"{self.base_url}/submission/mark_final"
            final_data = {
                "submission_id": submission_id
            }
            
            logger.info(f"Marking submission {submission_id} as final")
            response = self.session.post(final_url, json=final_data)
            
            if response.status_code == 200:
                logger.info(f"Successfully marked submission {submission_id} as final")
                return True
            else:
                logger.error(f"Mark final failed for {submission_id}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Mark final error for {submission_id}: {str(e)}")
            return False
    
    def list_submissions(self) -> Optional[List[Dict]]:
        """List all submissions for the current user"""
        try:
            list_url = f"{self.base_url}/submission/list"
            
            logger.info("Fetching submission list")
            response = self.session.get(list_url)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Raw API response: {result}")
                
                # The API returns submissions in the 'files' key
                submissions = result.get('files', [])
                
                logger.info(f"Found {len(submissions)} submissions")
                return submissions
            else:
                logger.error(f"List submissions failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"List submissions error: {str(e)}")
            return None
    
    def unmark_final(self, submission_id: str) -> bool:
        """Unmark a submission as final"""
        try:
            unmark_url = f"{self.base_url}/submission/unmark_final"
            unmark_data = {
                "submission_id": submission_id
            }
            
            logger.info(f"Unmarking submission {submission_id} as final")
            response = self.session.post(unmark_url, json=unmark_data)
            
            if response.status_code == 200:
                logger.info(f"Successfully unmarked submission {submission_id} as final")
                return True
            else:
                logger.error(f"Unmark final failed for {submission_id}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Unmark final error for {submission_id}: {str(e)}")
            return False
    
    def logout(self):
        """Clear authentication"""
        self.auth_token = None
        if 'Authorization' in self.session.headers:
            del self.session.headers['Authorization']
        logger.info("Logged out")

def load_credentials(creds_file: str = "creds.json") -> Dict:
    """Load bot credentials from JSON file"""
    try:
        with open(creds_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading credentials: {str(e)}")
        return {}

def load_bot_directories(bot_dirs_file: str = "bot_directories.txt") -> Dict[str, str]:
    """Load bot directory mappings from text file"""
    mappings = {}
    try:
        with open(bot_dirs_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse format: model_id : dir_path
                if ' : ' in line:
                    model_id, dir_path = line.split(' : ', 1)
                    mappings[model_id.strip()] = dir_path.strip()
        
        logger.info(f"Loaded {len(mappings)} bot directory mappings")
        return mappings
        
    except Exception as e:
        logger.error(f"Error loading bot directories: {str(e)}")
        return {}

def submit_bot(submitter: HuskyBenchSubmitter, model_id: str, username: str, password: str, 
               bot_dir: str, workspace_root: str) -> bool:
    """Submit a single bot to HuskyBench"""
    
    logger.info(f"Processing bot: {model_id}")
    
    # Login
    if not submitter.login(username, password):
        return False
    
    try:
        # First, list existing submissions and check for final ones
        submissions = submitter.list_submissions()
        if submissions is not None:
            logger.info("Current submissions:")
            final_submissions = []
            for i, submission in enumerate(submissions):
                submission_id = submission.get('id')
                is_final = submission.get('final', False)  # API uses 'final' not 'is_final'
                created_at = submission.get('created_at', 'Unknown')
                status_text = "FINAL" if is_final else "draft"
                
                logger.info(f"  {i+1}. ID: {submission_id} | Status: {status_text} | Created: {created_at}")
                
                if is_final:
                    final_submissions.append(submission_id)
            
            # Unmark any existing final submissions
            if final_submissions:
                logger.info(f"Found {len(final_submissions)} final submission(s). Unmarking them...")
                for final_id in final_submissions:
                    if not submitter.unmark_final(final_id):
                        logger.warning(f"Failed to unmark submission {final_id}, continuing anyway...")
            else:
                logger.info("No existing final submissions found.")
        else:
            logger.warning("Could not retrieve submission list, proceeding with upload...")
        
        # Construct file paths
        player_py_path = os.path.join(workspace_root, bot_dir, "player.py")
        requirements_txt_path = os.path.join(workspace_root, bot_dir, "requirements.txt")
        
        # Verify files exist
        if not os.path.exists(player_py_path):
            logger.error(f"player.py not found for {model_id} at {player_py_path}")
            return False
            
        if not os.path.exists(requirements_txt_path):
            logger.error(f"requirements.txt not found for {model_id} at {requirements_txt_path}")
            return False
        
        # Upload submission
        submission_id = submitter.upload_submission(player_py_path, requirements_txt_path)
        if not submission_id:
            return False
        
        # Mark as final
        if not submitter.mark_final(submission_id):
            return False
        
        logger.info(f"Successfully submitted bot {model_id} (submission ID: {submission_id})")
        return True
        
    finally:
        # Always logout after processing each bot
        submitter.logout()

def main():
    """Main submission process"""
    
    # Get workspace root
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    
    logger.info("Starting HuskyBench submission process")
    logger.info(f"Workspace root: {workspace_root}")
    
    # Load credentials and directory mappings
    credentials = load_credentials(os.path.join(workspace_root, "creds.json"))
    bot_directories = load_bot_directories(os.path.join(workspace_root, "bot_directories.txt"))
    
    if not credentials:
        logger.error("No credentials loaded. Exiting.")
        return
    
    if not bot_directories:
        logger.error("No bot directories loaded. Exiting.")
        return
    
    # Initialize submitter
    submitter = HuskyBenchSubmitter()
    
    # Track results
    successful_submissions = []
    failed_submissions = []
    
    # Process each bot that has both credentials and directory mapping
    for model_id, bot_dir in bot_directories.items():
        if model_id in credentials:
            creds = credentials[model_id]
            username = creds.get('username')
            password = creds.get('password')
            
            if username and password:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing: {model_id}")
                logger.info(f"Directory: {bot_dir}")
                logger.info(f"Username: {username}")
                
                if submit_bot(submitter, model_id, username, password, bot_dir, workspace_root):
                    successful_submissions.append(model_id)
                else:
                    failed_submissions.append(model_id)
            else:
                logger.warning(f"Invalid credentials for {model_id}")
                failed_submissions.append(model_id)
        else:
            logger.warning(f"No credentials found for {model_id}")
            failed_submissions.append(model_id)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUBMISSION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total bots processed: {len(bot_directories)}")
    logger.info(f"Successful submissions: {len(successful_submissions)}")
    logger.info(f"Failed submissions: {len(failed_submissions)}")
    
    if successful_submissions:
        logger.info(f"\nSuccessful submissions:")
        for model_id in successful_submissions:
            logger.info(f"  ✓ {model_id}")
    
    if failed_submissions:
        logger.info(f"\nFailed submissions:")
        for model_id in failed_submissions:
            logger.info(f"  ✗ {model_id}")

if __name__ == "__main__":
    main()
