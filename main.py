import sys
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from docker_service import DockerService
from llm.prompt_processor import OpenRouterPromptProcessor
from config import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_NUM_ITERATIONS


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    k_iterations: int = DEFAULT_NUM_ITERATIONS
    prompt_file: str = "prompt/generate.txt"


@dataclass
class IterationResult:
    """Result of a single iteration"""
    iteration: int
    bot_dir: Optional[str]
    success: bool
    error: str
    result: Optional[Dict[str, Any]]
    code: Optional[str] = None
    requirements: Optional[str] = None
    error_log: Optional[str] = None


class FeedbackAnalyzer:
    """Handles analysis and collection of feedback data from iterations"""
    
    @staticmethod
    def extract_bot_performance(game_logs: List[Dict]) -> Tuple[List[str], str]:
        """Extract bot player IDs and create performance summary"""
        bot_player_ids = []
        for game in game_logs:
            if 'botPlayerId' in game:
                bot_player_ids.append(game['botPlayerId'])
        
        bot_id_info = ""
        bot_performance_summary = ""
        
        if bot_player_ids:
            unique_bot_ids = list(set(bot_player_ids))
            bot_id_info = f"""
YOUR BOT PLAYER IDS: {unique_bot_ids}
When analyzing the game data below, focus on these player IDs as they represent YOUR bot's performance.
"""
            
            # Calculate bot performance metrics
            if game_logs:
                total_score = 0
                game_count = 0
                
                for game in game_logs:
                    if 'botPerformance' in game:
                        perf = game['botPerformance']
                        # Only use gameScore which represents delta gain per individual game
                        # Don't use finalMoney or finalDelta as they accumulate across games
                        total_score += perf.get('gameScore', 0)
                        game_count += 1
                
                if game_count > 0:
                    avg_score = total_score / game_count
                    
                    bot_performance_summary = f"""
YOUR BOT PERFORMANCE SUMMARY (across {game_count} games):
- Average Game Score (Delta per Game): {avg_score:.2f}
- Total Game Score: {total_score:.2f}
- Total Games Played: {game_count}

Performance Analysis:
- Positive scores indicate profit per game
- Negative scores indicate losses per game
- Focus on improving average game score for better performance
"""
        
        return bot_player_ids, bot_id_info, bot_performance_summary
    
    @staticmethod
    def collect_feedback_data(bot_dir: str, iteration: int) -> Dict[str, Any]:
        """Collect feedback data from the verified iteration directory"""
        feedback_data = {
            'success': False,
            'error_message': '',
            'errors': '',
            'game_logs': [],
            'validation_errors': ''
        }
        
        # Check if bot_dir exists
        if not bot_dir or not os.path.exists(bot_dir):
            feedback_data['error_message'] = f"Bot directory not found: {bot_dir}"
            return feedback_data
        
        # Use iteration-specific verified directory
        verified_dir = os.path.join(bot_dir, "verified", f"{iteration}_iteration")
        
        # Read current code from iteration-specific directory
        FeedbackAnalyzer._read_iteration_code(verified_dir, feedback_data, iteration)
        
        if not os.path.exists(verified_dir):
            feedback_data['error_message'] = f"Verified directory for iteration {iteration} not found: {verified_dir}"
            print(f"⚠️  Verified directory not found: {verified_dir}")
            # Still try to read game logs from bot_dir if verified_dir doesn't exist
            FeedbackAnalyzer._read_game_logs(bot_dir, feedback_data, iteration)
            return feedback_data
        
        # Read error log
        FeedbackAnalyzer._read_error_log(verified_dir, feedback_data)
        
        # Always read game logs - this is critical for feedback
        FeedbackAnalyzer._read_game_logs(verified_dir, feedback_data, iteration)
        
        print(f"✅ Feedback data collection completed for iteration {iteration}")
        return feedback_data
    
    @staticmethod
    def _read_iteration_code(verified_dir: str, feedback_data: Dict[str, Any], iteration: int) -> None:
        """Read iteration-specific code files"""
        try:
            player_path = os.path.join(verified_dir, "player.py")
            requirements_path = os.path.join(verified_dir, "requirements.txt")
            
            if os.path.exists(player_path):
                with open(player_path, 'r', encoding='utf-8') as f:
                    feedback_data['current_code'] = f.read()
            
            if os.path.exists(requirements_path):
                with open(requirements_path, 'r', encoding='utf-8') as f:
                    feedback_data['current_requirements'] = f.read()
        except Exception as e:
            print(f"Warning: Could not read iteration {iteration} code for feedback: {e}")
    
    @staticmethod
    def _read_error_log(verified_dir: str, feedback_data: Dict[str, Any]) -> None:
        """Read and parse error log"""
        error_log_path = os.path.join(verified_dir, "error.log")
        if os.path.exists(error_log_path):
            # Add a small delay to ensure file is fully written
            import time
            time.sleep(0.5)
            
            try:
                with open(error_log_path, 'r', encoding='utf-8') as f:
                    error_content = f.read()
                    
                    # Check if there are any actual errors (not just "No errors detected")
                    if "No errors detected" in error_content and len(error_content.strip()) < 200:
                        # Only mark as success if the log is very short and explicitly says no errors
                        feedback_data['success'] = True
                    else:
                        # If there's substantial content, treat as having errors
                        feedback_data['errors'] = error_content
                        
                        # Extract error lines with context (10 lines after each error)
                        error_lines_with_context = FeedbackAnalyzer._extract_error_lines_with_context(error_content)
                        if error_lines_with_context:
                            feedback_data['error_lines_with_context'] = error_lines_with_context
                        else:
                            feedback_data['error_lines_with_context'] = None
                            feedback_data['errors'] = None
                        # Extract validation errors specifically
                        if "Code validation failed" in error_content:
                            lines = error_content.split('\n')
                            validation_lines = [line for line in lines 
                                              if 'validation' in line.lower() or 
                                                 'syntax' in line.lower() or 
                                                 'import' in line.lower()]
                            feedback_data['validation_errors'] = '\n'.join(validation_lines)
                        
                        # Extract poker client log errors
                        if "POKER CLIENT LOGS:" in error_content:
                            poker_log_section = error_content.split("POKER CLIENT LOGS:")[1]
                            if "Poker Client Log:" in poker_log_section:
                                poker_log_content = poker_log_section.split("Poker Client Log:")[1].split("\n\n")[0]
                                if poker_log_content.strip():
                                    # Only include poker client logs if they contain actual errors
                                    poker_log_lines = poker_log_content.lower()
                                    if any(error_keyword in poker_log_lines for error_keyword in ['error', 'exception', 'failed', 'timeout', 'invalid', 'syntax']):
                                        feedback_data['poker_client_errors'] = poker_log_content.strip()
                                        print(f"🎯 Found poker client errors in iteration log")
                                    else:
                                        print(f"📝 Poker client logs found but no errors detected - skipping inclusion in feedback")
            except Exception as e:
                print(f"⚠️  Error reading error log: {e}")
                feedback_data['errors'] = f"Error reading error log: {str(e)}"
        else:
            # No error log file found, assume there might be issues
            feedback_data['errors'] = f"Error log file not found for iteration in {verified_dir}"
    
    @staticmethod
    def _extract_error_lines_with_context(error_content: str) -> str:
        """Extract error lines with 10 lines of context after each error"""
        lines = error_content.split('\n')
        error_sections = []
        
        for i, line in enumerate(lines):
            # Look for lines that contain error indicators
            if any(keyword in line.lower() and "detected" not in line.lower() for keyword in ['error', 'exception', 'failed', 'timeout', 'invalid', 'syntax']):
                # Get the error line and 10 lines after it
                error_section = [line]  # Start with the error line
                
                # Add up to 10 lines after the error
                for j in range(1, 11):
                    if i + j < len(lines):
                        error_section.append(lines[i + j])
                    else:
                        break
                
                # Join the lines and add to sections
                error_sections.append('\n'.join(error_section))
        
        # Join all error sections with separators
        if error_sections:
            return '\n\n--- NEXT ERROR ---\n\n'.join(error_sections)
        else:
            return ""
    
    @staticmethod
    def _read_game_logs(verified_dir: str, feedback_data: Dict[str, Any], iteration: int) -> None:
        """Read and parse game log files"""
        try:
            if not os.path.exists(verified_dir):
                print(f"⚠️  Verified directory not found for iteration {iteration}: {verified_dir}")
                return
                
            game_log_files = [f for f in os.listdir(verified_dir) 
                            if f.startswith("gamelog_") and f.endswith(".json")]
            print(f"🎮 Found {len(game_log_files)} game log files in iteration {iteration}: {game_log_files}")
            
            if not game_log_files:
                print(f"📝 No game log files found in iteration {iteration} directory")
                return
            
            for log_file in sorted(game_log_files):
                log_path = os.path.join(verified_dir, log_file)
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        game_data = json.load(f)
                        
                        # Extract key performance metrics
                        game_summary = FeedbackAnalyzer._extract_game_summary(game_data)
                        feedback_data['game_logs'].append(game_summary)
                        print(f"✅ Successfully read game log: {log_file}")
                        
                except Exception as e:
                    print(f"⚠️  Error reading game log {log_file}: {str(e)}")
                    feedback_data['errors'] += f"\nError reading game log {log_file}: {str(e)}"
            
            print(f"🎯 Total game logs loaded for iteration {iteration}: {len(feedback_data['game_logs'])}")
                    
        except Exception as e:
            print(f"⚠️  Error listing game log files in iteration {iteration}: {str(e)}")
            feedback_data['errors'] += f"\nError listing game log files: {str(e)}"
    
    @staticmethod
    def _extract_game_summary(game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key performance metrics from game data"""
        game_summary = {
            'gameId': game_data.get('gameId', 'unknown'),
            'players': game_data.get('usernameMapping', {}),
            'playerNames': game_data.get('playerNames', {}),
            'finalMoney': game_data.get('playerMoney', {}).get('finalMoney', {}),
            'finalDelta': game_data.get('playerMoney', {}).get('finalDelta', {}),
            'gameScores': game_data.get('playerMoney', {}).get('gameScores', {}),
            'rounds': len(game_data.get('rounds', {})),
            'finalBoard': game_data.get('finalBoard', []),
            'playerHands': game_data.get('playerHands', {}),
            'blinds': game_data.get('blinds', {})
        }
        
        # Extract bot player ID for this game
        username_mapping = game_data.get('usernameMapping', {})
        bot_player_id = None
        for username, player_id in username_mapping.items():
            if 'test_client' in username or 'iter' in username:
                bot_player_id = player_id
                break
        
        if bot_player_id:
            game_summary['botPlayerId'] = bot_player_id
            game_summary['botPerformance'] = {
                'finalMoney': game_data.get('playerMoney', {}).get('finalMoney', {}).get(str(bot_player_id), 0),
                'finalDelta': game_data.get('playerMoney', {}).get('finalDelta', {}).get(str(bot_player_id), 0),
                'gameScore': game_data.get('playerMoney', {}).get('gameScores', {}).get(str(bot_player_id), 0)
            }
        
        return game_summary


class PromptBuilder:
    """Handles creation of feedback prompts for iterative generation"""
    
    @staticmethod
    def create_feedback_prompt(original_prompt: str, iteration: int, feedback_data: Dict[str, Any], 
                              previous_code: Optional[str] = None, 
                              previous_requirements: Optional[str] = None) -> str:
        """Create an improved prompt based on feedback from previous iteration"""
        
        feedback_prompt = PromptBuilder._build_base_prompt(original_prompt, iteration, feedback_data)
        
        # Include previous iteration's code
        feedback_prompt += PromptBuilder._include_previous_code(
            feedback_data, previous_code, previous_requirements
        )
        
        # Include errors
        if feedback_data.get('errors'):
            feedback_prompt += PromptBuilder._include_errors(feedback_data['errors'])
        
        # Include error lines with context
        if feedback_data.get('error_lines_with_context'):
            feedback_prompt += PromptBuilder._include_error_lines_with_context(feedback_data['error_lines_with_context'])
        
        # Include game performance data
        if feedback_data.get('game_logs'):
            feedback_prompt += PromptBuilder._include_game_performance(feedback_data['game_logs'])
        
        # Include validation errors
        if feedback_data.get('validation_errors'):
            feedback_prompt += PromptBuilder._include_validation_errors(feedback_data['validation_errors'])
        
        # Include poker client errors
        if feedback_data.get('poker_client_errors'):
            feedback_prompt += PromptBuilder._include_poker_client_errors(feedback_data['poker_client_errors'])
        
        # Add improvement instructions
        feedback_prompt += PromptBuilder._add_improvement_instructions(iteration)
        
        return feedback_prompt
    
    @staticmethod
    def _build_base_prompt(original_prompt: str, iteration: int, feedback_data: Dict[str, Any]) -> str:
        """Build the base feedback prompt"""
        return f"""
{original_prompt}

ITERATION {iteration} FEEDBACK AND IMPROVEMENT REQUEST:

You are now working on iteration {iteration}. Based on the previous iteration results and code, please improve the implementation.

PREVIOUS RESULTS:
- Success: {feedback_data.get('success', False)}
- Error Message: {feedback_data.get('error_message', 'None')}

"""
    
    @staticmethod
    def _include_previous_code(feedback_data: Dict[str, Any], previous_code: Optional[str], 
                              previous_requirements: Optional[str]) -> str:
        """Include previous iteration's code in the prompt"""
        if previous_code:
            return f"""
PREVIOUS ITERATION CODE (player.py):
```python
{previous_code}
```

PREVIOUS ITERATION REQUIREMENTS (requirements.txt):
```text
{previous_requirements or '# No requirements specified'}
```

"""
        elif feedback_data.get('current_code'):
            return f"""
CURRENT CODE (player.py):
```python
{feedback_data['current_code']}
```

CURRENT REQUIREMENTS (requirements.txt):
```text
{feedback_data.get('current_requirements', '# No requirements specified')}
```

"""
        return ""
    
    @staticmethod
    def _include_errors(errors: str) -> str:
        """Include error information in the prompt"""
        return f"""
ERRORS ENCOUNTERED:
{errors}

Please fix these specific errors in your implementation.
"""
    
    @staticmethod
    def _include_error_lines_with_context(error_lines_with_context: str) -> str:
        """Include error lines with context in the prompt"""
        return f"""
ERROR LINES WITH CONTEXT:
{error_lines_with_context}

Please investigate these specific error lines in your code.
"""
    
    @staticmethod
    def _include_game_performance(game_logs: List[Dict[str, Any]]) -> str:
        """Include game performance data in the prompt"""
        bot_player_ids, bot_id_info, bot_performance_summary = FeedbackAnalyzer.extract_bot_performance(game_logs)
        
        return f"""
GAME PERFORMANCE DATA:
{bot_id_info}
{bot_performance_summary}
{json.dumps(game_logs, indent=2)}

IMPORTANT: When analyzing the game performance data above, focus on the player ID that corresponds to YOUR bot implementation. 
Look for your bot's performance in the 'gameScores' field, which shows the delta gain/loss for each individual game.
Your bot will be identified by a player ID (usually a number) in the game data.

PERFORMANCE METRICS EXPLANATION:
- 'gameScores': Shows the profit/loss for each individual game (this is what matters most)
- Positive gameScore = Profit in that game
- Negative gameScore = Loss in that game
- Focus on improving your strategy to achieve positive gameScores consistently

Please analyze YOUR bot's gameScore performance and improve the strategy based on these results.
"""
    
    @staticmethod
    def _include_validation_errors(validation_errors: str) -> str:
        """Include validation errors in the prompt"""
        return f"""
CODE VALIDATION ERRORS:
{validation_errors}

Please fix these validation issues in your code.
"""
    
    @staticmethod
    def _include_poker_client_errors(poker_client_errors: str) -> str:
        """Include poker client errors in the prompt"""
        return f"""
POKER CLIENT ERRORS:
{poker_client_errors}

Please investigate these errors in your poker client logs and fix any issues in your implementation.
"""
    
    @staticmethod
    def _add_improvement_instructions(iteration: int) -> str:
        """Add improvement instructions to the prompt"""
        return f"""

IMPROVEMENT INSTRUCTIONS:
1. You are on iteration {iteration} - build upon the previous code above
2. Analyze the errors and game performance data above
3. Fix any syntax, import, or runtime errors from the previous iteration
4. Improve the poker strategy based on game results
5. Ensure the code follows the exact template requirements
6. Return the improved implementation in the same format (Python code block + requirements.txt block)

Focus on making the bot more competitive and error-free. You can see exactly what was wrong with the previous version and should fix those specific issues.
"""


class IterativeGenerator:
    """Handles iterative generation with feedback loops"""
    
    def __init__(self, processor: OpenRouterPromptProcessor):
        self.processor = processor
        self.feedback_analyzer = FeedbackAnalyzer()
        self.prompt_builder = PromptBuilder()
    
    def run_iterations(self, selected_model: Dict[str, Any], original_prompt: str, 
                      config: ProcessingConfig) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Run iterative generation for k iterations with feedback"""
        
        print(f"\n🔄 Starting iterative generation for {config.k_iterations} iterations...")
        
        current_prompt = original_prompt
        best_result = None
        best_bot_dir = None
        all_results: List[IterationResult] = []
        
        # Create a single bot directory for all iterations
        main_bot_dir = None
        
        # Track previous iteration's code for feedback
        previous_code = None
        previous_requirements = None

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean model name for directory - remove/replace invalid characters
        model_name = selected_model["id"].replace("/", "_").replace("-", "_").replace(":", "_").replace("\\", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
        
        # Create directory name
        dir_name = f"{model_name}_{timestamp}"
        dir_path = os.path.join("bot", dir_name)
        
        # Create directory
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created new directory: {dir_path}")

        main_bot_dir = dir_path

        port = self.processor.docker_service.generate_random_port()
        try:
            for iteration in range(1, config.k_iterations + 1):
                print(f"\n{'='*60}")
                print(f"🔄 ITERATION {iteration}/{config.k_iterations}")
                print(f"{'='*60}")
                
                # Debug: Show current main_bot_dir
                print(f"📁 Using main bot directory: {main_bot_dir}")
                
                # Process the current iteration
                iteration_result = self._process_single_iteration(
                    iteration, config.k_iterations, selected_model, current_prompt, 
                    config, main_bot_dir, previous_code, previous_requirements, port=port
                )
                
                all_results.append(iteration_result)
                
                # Debug: Show current main_bot_dir status
                print(f"📁 Main bot directory: {main_bot_dir}")
                
                if iteration_result.success:
                    best_result = iteration_result.result
                    best_bot_dir = iteration_result.bot_dir
                    
                    # If successful and not the last iteration, create improvement prompt
                    if iteration < config.k_iterations:
                        try:
                            current_prompt = self._create_next_iteration_prompt(
                                original_prompt, iteration, iteration_result, previous_code, previous_requirements
                            )
                        except Exception as e:
                            print(f"⚠️  Error creating feedback prompt: {e}")
                            # Fallback to original prompt
                            current_prompt = original_prompt
                else:
                    # Even if iteration failed, create feedback prompt for next iteration
                    if iteration < config.k_iterations:
                        try:
                            current_prompt = self._create_next_iteration_prompt(
                                original_prompt, iteration, iteration_result, previous_code, previous_requirements
                            )
                        except Exception as e:
                            print(f"⚠️  Error creating feedback prompt: {e}")
                            # Fallback to original prompt
                            current_prompt = original_prompt
                
                # Update previous code for next iteration
                previous_code = iteration_result.code
                previous_requirements = iteration_result.requirements
                
                # Debug: Show iteration completion
                print(f"✅ Iteration {iteration} completed. Success: {iteration_result.success}")
                if iteration < config.k_iterations:
                    print(f"🔄 Preparing for iteration {iteration + 1}...")
                else:
                    print("🏁 All iterations completed!")
            
            # Display summary
            self._display_iteration_summary(config.k_iterations, main_bot_dir, all_results, best_result, best_bot_dir)
            
            # Create summary log in main bot directory
            if main_bot_dir:
                self._create_summary_log(main_bot_dir, config.k_iterations, all_results, selected_model)
            
            return best_result, best_bot_dir
        finally:
            self.processor.docker_service.release_port(port)
            self.processor.docker_service.cleanup_containers_by_port(port)
    
    def _process_single_iteration(self, iteration: int, total_iterations: int, 
                                 selected_model: Dict[str, Any], current_prompt: str,
                                 config: ProcessingConfig, main_bot_dir: Optional[str],
                                 previous_code: Optional[str], previous_requirements: Optional[str],
                                 port: Optional[int] = None) -> IterationResult:
        """Process a single iteration"""
        
        print(f"🚀 Processing prompt with {selected_model['id']}...")
        print("⏳ This may take a moment...")
        
        try:
            # Process the prompt
            result = self.processor.process_prompt(
                model_id=selected_model['id'],
                prompt=current_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Save to main bot directory
            bot_output_file = os.path.join(main_bot_dir, f"output_iteration_{iteration}.log")
            self.processor.save_result_to_log(result, selected_model['id'], current_prompt, bot_output_file, save_code_blocks=False)
            print(f"📄 Output log also saved to: {bot_output_file}")
            
            # Extract response content
            if "choices" in result and len(result["choices"]) > 0:
                response_content = result["choices"][0]["message"]["content"]
                python_code, text_content = self.processor.extract_code_blocks(response_content)

                if not python_code:
                    return IterationResult(
                        iteration=iteration,
                        bot_dir=main_bot_dir,
                        success=False,
                        error="No code blocks found in response",
                        result=result,
                        code=python_code,
                        requirements=text_content,
                        error_log=None
                    )
                
                self._save_and_test_code(
                    iteration, selected_model, python_code, text_content, main_bot_dir, port=port
                )

                error_log_content = ""
                error_log_path = os.path.join(main_bot_dir, "verified", f"{iteration}_iteration", "error.log")
                if os.path.exists(error_log_path):
                    with open(error_log_path, 'r', encoding='utf-8') as f:
                        error_log_content = f.read()
                else:
                    print(f"❌ Error: error.log not found in {main_bot_dir}")

                # Extract error contexts from error log
                error_contexts = []
                if error_log_content:
                    lines = error_log_content.split('\n')
                    for i, line in enumerate(lines):
                        if 'error' in line.lower() and 'detected' not in line.lower():
                            # Get 10 lines before and after the error line
                            start = max(0, i - 10)
                            end = min(len(lines), i + 11)
                            context = '\n'.join(lines[start:end])
                            error_contexts.append(context)
                
                if error_contexts:
                    error_contexts_str = "\n".join(error_contexts)
                else:
                    error_contexts_str = "No error contexts found in error.log"

                if len(error_contexts) > 0:
                    return IterationResult(
                        iteration=iteration,
                        bot_dir=main_bot_dir,
                        success=False,
                        error=error_contexts_str,
                        result=result,
                        code=python_code,
                        requirements=text_content,
                        error_log=error_log_content
                    )
                else:
                    return IterationResult(
                        iteration=iteration,
                        bot_dir=main_bot_dir,
                        success=True,
                        error="",
                        result=result,
                        code=python_code,
                        requirements=text_content,
                        error_log="No error detected"
                    )
                
        except Exception as e:
            print(f"❌ Error in iteration {iteration}: {e}")
            return IterationResult(
                iteration=iteration,
                bot_dir=main_bot_dir,
                success=False,
                error=str(e),
                result=None,
                error_log=None
            )
    
    def _save_and_test_code(self, iteration: int, selected_model: Dict[str, Any], 
                           python_code: str, text_content: str, main_bot_dir: Optional[str],
                           port: Optional[int] = None) -> Optional[str]:
        """Save code and test it"""
        
        print(f"🔧 Saving code for iteration {iteration}")
        print(f"📁 Main bot directory: {main_bot_dir}")
        
        if not main_bot_dir:
            print("❌ Error: main_bot_dir is required but not provided")
            return None
        
        # Always use the main_bot_dir for all iterations
        print(f"🔄 Using main bot directory: {main_bot_dir}")
        
        # Save code directly to main_bot_dir (not creating new directories)
        try:
            # Save player.py to main directory
            player_path = os.path.join(main_bot_dir, "player.py")
            with open(player_path, 'w', encoding='utf-8') as f:
                f.write(python_code)
            print(f"✅ Python code saved to: {player_path}")
            
            # Save requirements.txt to main directory
            requirements_path = os.path.join(main_bot_dir, "requirements.txt")
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"✅ Requirements saved to: {requirements_path}")
            
            bot_dir = main_bot_dir
            
        except Exception as e:
            print(f"❌ Error saving code to main directory: {e}")
            return None
        
        print(f"📁 Final bot directory: {bot_dir}")
        
        # Save iteration-specific code
        if bot_dir:
            self._save_iteration_specific_code(bot_dir, iteration, python_code, text_content)
            
            # Test the generated code
            if self.processor.docker_service:
                test_success, test_error = self.processor.test_generated_code(
                    bot_dir, f"{selected_model['id']}_iter_{iteration}", iteration, port=port
                )
                
                if test_success:
                    print("✅ Game test completed successfully!")
                else:
                    print(f"⚠️  Game test failed: {test_error}")
                
                return bot_dir if test_success else None
            else:
                print("⚠️  Docker service not available - skipping game test")
                return bot_dir
        
        return None
    
    def _save_iteration_specific_code(self, bot_dir: str, iteration: int, python_code: str, text_content: str) -> None:
        """Save iteration-specific code to iteration directory"""
        iteration_dir = os.path.join(bot_dir, "verified", f"{iteration}_iteration")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save player.py to iteration directory
        iteration_player_path = os.path.join(iteration_dir, "player.py")
        with open(iteration_player_path, 'w', encoding='utf-8') as f:
            f.write(python_code)
        print(f"✅ Iteration {iteration} Python code saved to: {iteration_player_path}")
        
        # Save requirements.txt to iteration directory
        iteration_requirements_path = os.path.join(iteration_dir, "requirements.txt")
        with open(iteration_requirements_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"✅ Iteration {iteration} requirements saved to: {iteration_requirements_path}")
    
    def _create_next_iteration_prompt(self, original_prompt: str, iteration: int, 
                                     iteration_result: IterationResult, previous_code: Optional[str], 
                                     previous_requirements: Optional[str]) -> str:
        """Create prompt for next iteration based on current results"""
        
        if iteration_result.success and iteration_result.bot_dir:
            # Iteration succeeded, collect feedback from the bot directory
            feedback_data = self.feedback_analyzer.collect_feedback_data(iteration_result.bot_dir, iteration)
            feedback_data['success'] = True
            print("📈 Creating improvement prompt based on successful game results...")
            
            # Print feedback data summary
            self._print_feedback_data_summary(feedback_data, iteration)
        else:
            # Iteration failed or no bot directory, create basic feedback data
            error_lines_with_context = ""
            
            # Read error.log from current iteration if it exists
            if iteration_result.bot_dir:
                current_iteration_dir = os.path.join(iteration_result.bot_dir, "verified", f"{iteration}_iteration")
                current_error_log_path = os.path.join(current_iteration_dir, "error.log")
                if os.path.exists(current_error_log_path):
                    # Add a small delay to ensure file is fully written
                    import time
                    time.sleep(0.5)
                    
                    try:
                        with open(current_error_log_path, 'r', encoding='utf-8') as f:
                            current_error_content = f.read()
                            error_lines_with_context = FeedbackAnalyzer._extract_error_lines_with_context(current_error_content)
                            
                            # Always include current iteration content, even if no errors
                            if error_lines_with_context:
                                print(f"📖 Read error log from current iteration {iteration}")
                            else:
                                # No errors in current iteration, include the full content as context
                                error_lines_with_context = f"CURRENT ITERATION CONTEXT (NO ERRORS):\n{current_error_content}"
                                print(f"📖 Read current iteration context (no errors) from iteration {iteration}")
                    except Exception as e:
                        print(f"⚠️  Error reading current iteration error log: {e}")
                        error_lines_with_context = f"Error reading error log: {str(e)}"
            
            # If no current error log, try iteration_result.error_log
            if not error_lines_with_context and iteration_result.error_log:
                error_lines_with_context = FeedbackAnalyzer._extract_error_lines_with_context(iteration_result.error_log)
            
            feedback_data = {
                'success': False,
                'error_message': iteration_result.error,
                'errors': iteration_result.error_log if iteration_result.error_log else iteration_result.error,
                'error_lines_with_context': error_lines_with_context,
                'game_logs': [],
                'validation_errors': ''
            }
            
            print("🔧 Creating correction prompt based on errors...")
            
            # Print feedback data summary
            self._print_feedback_data_summary(feedback_data, iteration)
        
        # Always try to read game logs from current iteration directory when JSON files exist
        # This applies to both successful and failed iterations
        if iteration_result.bot_dir:
            current_iteration_dir = os.path.join(iteration_result.bot_dir, "verified", f"{iteration}_iteration")
            # Check if JSON files exist in the iteration directory
            if os.path.exists(current_iteration_dir):
                json_files = [f for f in os.listdir(current_iteration_dir) if f.endswith('.json')]
                if json_files:
                    print(f"🎮 Found {len(json_files)} JSON game log files - including game logs in feedback")
                    self.feedback_analyzer._read_game_logs(current_iteration_dir, feedback_data, iteration)
                else:
                    print(f"📝 No JSON game log files found in iteration {iteration} directory")
        
        return self.prompt_builder.create_feedback_prompt(
            original_prompt, iteration + 1, feedback_data, previous_code, previous_requirements
        )
    
    def _print_feedback_data_summary(self, feedback_data: Dict[str, Any], iteration: int) -> None:
        """Print a summary of the feedback data that was created"""
        print(f"\n📊 FEEDBACK DATA SUMMARY FOR ITERATION {iteration}:")
        print("=" * 60)
        
        # Basic info
        print(f"✅ Success: {feedback_data.get('success', False)}")
        print(f"⚠️  Error Message: {feedback_data.get('error_message', 'None')}")
        
        # Error information
        errors = feedback_data.get('errors', '')
        if errors:
            print(f"❌ Errors: {len(errors)} characters")
            if len(errors) > 200:
                print(f"   Preview: {errors[:200]}...")
            else:
                print(f"   Content: {errors}")
        else:
            print("❌ Errors: None")
        
        # Error lines with context
        error_lines = feedback_data.get('error_lines_with_context', '')
        if error_lines:
            print(f"🔍 Error Lines with Context: {len(error_lines)} characters")
            if len(error_lines) > 200:
                print(f"   Preview: {error_lines[:200]}...")
            else:
                print(f"   Content: {error_lines}")
        else:
            print("🔍 Error Lines with Context: None")
        
        # Game logs
        game_logs = feedback_data.get('game_logs', [])
        print(f"🎮 Game Logs: {len(game_logs)} entries")
        for i, log in enumerate(game_logs[:3]):  # Show first 3
            game_score = log.get('botPerformance', {}).get('gameScore', 'N/A')
            print(f"   Game {i+1}: {log.get('gameId', 'Unknown')} - Game Score (Delta): {game_score}")
        if len(game_logs) > 3:
            print(f"   ... and {len(game_logs) - 3} more games")
        
        # Validation errors
        validation_errors = feedback_data.get('validation_errors', '')
        if validation_errors:
            print(f"🔧 Validation Errors: {len(validation_errors)} characters")
            if len(validation_errors) > 200:
                print(f"   Preview: {validation_errors[:200]}...")
            else:
                print(f"   Content: {validation_errors}")
        else:
            print("🔧 Validation Errors: None")
        
        # Poker client errors
        poker_errors = feedback_data.get('poker_client_errors', '')
        if poker_errors:
            print(f"🎯 Poker Client Errors: {len(poker_errors)} characters")
            if len(poker_errors) > 200:
                print(f"   Preview: {poker_errors[:200]}...")
            else:
                print(f"   Content: {poker_errors}")
        else:
            print("🎯 Poker Client Errors: None")
        
        # Current code info
        current_code = feedback_data.get('current_code', '')
        if current_code:
            print(f"💻 Current Code: {len(current_code)} characters")
            lines = current_code.count('\n') + 1
            print(f"   Lines: {lines}")
        else:
            print("💻 Current Code: None")
        
        # Current requirements info
        current_requirements = feedback_data.get('current_requirements', '')
        if current_requirements:
            print(f"📦 Current Requirements: {len(current_requirements)} characters")
            lines = current_requirements.count('\n') + 1
            print(f"   Lines: {lines}")
        else:
            print("📦 Current Requirements: None")
        
        print("=" * 60)
    
    def _display_iteration_summary(self, k_iterations: int, main_bot_dir: Optional[str], 
                                  all_results: List[IterationResult], best_result: Optional[Dict[str, Any]], 
                                  best_bot_dir: Optional[str]) -> None:
        """Display summary of all iterations"""
        
        print(f"\n{'='*60}")
        print("🎯 ITERATIVE GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total iterations: {k_iterations}")
        
        if main_bot_dir:
            self._display_directory_structure(main_bot_dir)
        
        # Show results for each iteration
        for result in all_results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"Iteration {result.iteration}: {status}")
            if not result.success:
                print(f"  ⚠️  Error: {result.error}")
        
        if best_result and best_bot_dir:
            self._display_best_result(best_bot_dir)
        else:
            print("\n⚠️  No successful implementation generated")
    
    def _display_directory_structure(self, main_bot_dir: str) -> None:
        """Display the verified directory structure"""
        print(f"📁 Bot directory: {main_bot_dir}")
        print("📂 Verified directory structure:")
        verified_dir = os.path.join(main_bot_dir, "verified")
        if os.path.exists(verified_dir):
            for item in sorted(os.listdir(verified_dir)):
                if os.path.isdir(os.path.join(verified_dir, item)):
                    print(f"   📁 {item}/")
                    iteration_dir = os.path.join(verified_dir, item)
                    for file in sorted(os.listdir(iteration_dir)):
                        file_type = "📄" if file.endswith(('.py', '.txt', '.log')) else "📄"
                        print(f"      {file_type} {file}")
    
    def _display_best_result(self, best_bot_dir: str) -> None:
        """Display information about the best result"""
        verified_dir = os.path.join(best_bot_dir, "verified")
        if os.path.exists(verified_dir):
            successful_iterations = []
            for item in os.listdir(verified_dir):
                if item.endswith('_iteration'):
                    iteration_num = int(item.split('_')[0])
                    iteration_dir = os.path.join(verified_dir, item)
                    if os.path.exists(os.path.join(iteration_dir, "player.py")):
                        successful_iterations.append(iteration_num)
            
            if successful_iterations:
                latest_iteration = max(successful_iterations)
                latest_code_path = os.path.join(verified_dir, f"{latest_iteration}_iteration", "player.py")
                print(f"\n✅ Best result in: {best_bot_dir}")
                print(f"🏆 Latest successful code: {latest_code_path}")
                print("🎉 Successfully generated and tested poker bot implementation!")
            else:
                print(f"\n✅ Best result in: {best_bot_dir}")
                print("🏆 Successfully generated and tested poker bot implementation!")
        else:
            print(f"\n✅ Best result in: {best_bot_dir}")
            print("🏆 Successfully generated and tested poker bot implementation!")
    
    def _create_summary_log(self, main_bot_dir: str, k_iterations: int, all_results: List[IterationResult], selected_model: Dict[str, Any]) -> None:
        """Create a summary log file in the main bot directory"""
        try:
            summary_log_path = os.path.join(main_bot_dir, "iteration_summary.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(summary_log_path, 'w', encoding='utf-8') as f:
                f.write(f"Iterative Generation Summary - {timestamp}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Model: {selected_model['id']}\n")
                f.write(f"Total Iterations: {k_iterations}\n")
                f.write(f"Bot Directory: {main_bot_dir}\n\n")
                
                f.write("ITERATION RESULTS:\n")
                f.write("-" * 20 + "\n")
                
                for result in all_results:
                    status = "SUCCESS" if result.success else "FAILED"
                    f.write(f"Iteration {result.iteration}: {status}\n")
                    if not result.success:
                        f.write(f"  Error: {result.error}\n")
                    f.write("\n")
                
                # Count successful iterations
                successful_count = sum(1 for result in all_results if result.success)
                f.write(f"SUMMARY:\n")
                f.write(f"- Successful iterations: {successful_count}/{k_iterations}\n")
                f.write(f"- Failed iterations: {k_iterations - successful_count}/{k_iterations}\n")
                
                if successful_count > 0:
                    f.write(f"- Success rate: {(successful_count/k_iterations)*100:.1f}%\n")
                    f.write(f"- Latest successful iteration: {max([r.iteration for r in all_results if r.success])}\n")
                else:
                    f.write(f"- Success rate: 0%\n")
                    f.write(f"- No successful iterations\n")
                
                f.write(f"\nFILES IN THIS DIRECTORY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"- player.py: Latest generated code\n")
                f.write(f"- requirements.txt: Latest requirements\n")
                f.write(f"- output_iteration_X.log: Individual iteration logs\n")
                f.write(f"- verified/X_iteration/: Iteration-specific files\n")
                f.write(f"- iteration_summary.log: This summary file\n")
            
            print(f"📄 Summary log created: {summary_log_path}")
            
        except Exception as e:
            print(f"⚠️  Error creating summary log: {e}")


class PromptProcessorApp:
    """Main application class for the prompt processor"""
    
    def __init__(self):
        self.docker_service = DockerService()
        self.processor = OpenRouterPromptProcessor(docker_service=self.docker_service)
        self.iterative_generator = IterativeGenerator(self.processor)
    
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
        processor = OpenRouterPromptProcessor(docker_service=docker_service)
        iterative_generator = IterativeGenerator(processor)
        
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
        print("🤖 OpenRouter Prompt Processor")
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
    """Main function to run the prompt processor"""
    app = PromptProcessorApp()
    app.run()


if __name__ == "__main__":
    main() 