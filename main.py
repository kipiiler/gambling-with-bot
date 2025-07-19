import sys
import json
import os
from docker_service import DockerService
docker_service = DockerService()
from llm.prompt_processor import OpenRouterPromptProcessor


def create_feedback_prompt(original_prompt: str, iteration: int, feedback_data: dict, previous_code: str = None, previous_requirements: str = None) -> str:
    """Create an improved prompt based on feedback from previous iteration"""
    
    feedback_prompt = f"""
{original_prompt}

ITERATION {iteration} FEEDBACK AND IMPROVEMENT REQUEST:

You are now working on iteration {iteration}. Based on the previous iteration results and code, please improve the implementation.

PREVIOUS RESULTS:
- Success: {feedback_data.get('success', False)}
- Error Message: {feedback_data.get('error_message', 'None')}

"""
    
    # Include previous iteration's code if available
    if previous_code:
        feedback_prompt += f"""
PREVIOUS ITERATION CODE (player.py):
```python
{previous_code}
```

PREVIOUS ITERATION REQUIREMENTS (requirements.txt):
```text
{previous_requirements or 'No requirements specified'}
```

"""
    elif feedback_data.get('current_code'):
        # Fallback to current code from bot directory
        feedback_prompt += f"""
CURRENT CODE (player.py):
```python
{feedback_data['current_code']}
```

CURRENT REQUIREMENTS (requirements.txt):
```text
{feedback_data.get('current_requirements', 'No requirements specified')}
```

"""
    
    if feedback_data.get('errors'):
        feedback_prompt += f"""
ERRORS ENCOUNTERED:
{feedback_data['errors']}

Please fix these specific errors in your implementation.
"""
    
    if feedback_data.get('game_logs'):
        # Extract bot player IDs from all games
        bot_player_ids = []
        for game in feedback_data['game_logs']:
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
            
            # Create a performance summary
            total_money = 0
            total_delta = 0
            total_score = 0
            game_count = 0
            
            for game in feedback_data['game_logs']:
                if 'botPerformance' in game:
                    perf = game['botPerformance']
                    total_money += perf.get('finalMoney', 0)
                    total_delta += perf.get('finalDelta', 0)
                    total_score += perf.get('gameScore', 0)
                    game_count += 1
            
            if game_count > 0:
                avg_money = total_money / game_count
                avg_delta = total_delta / game_count
                avg_score = total_score / game_count
                
                bot_performance_summary = f"""
YOUR BOT PERFORMANCE SUMMARY (across {game_count} games):
- Average Final Money: {avg_money:.2f}
- Average Money Delta: {avg_delta:.2f}
- Average Game Score: {avg_score:.2f}
- Total Games Played: {game_count}

Use this performance data to identify areas for improvement in your strategy.
"""
        
        feedback_prompt += f"""
GAME PERFORMANCE DATA:
{bot_id_info}
{bot_performance_summary}
{json.dumps(feedback_data['game_logs'], indent=2)}

IMPORTANT: When analyzing the game performance data above, focus on the player ID that corresponds to YOUR bot implementation. 
Look for your bot's performance in the 'players', 'finalMoney', 'finalDelta', and 'gameScores' fields.
Your bot will be identified by a player ID (usually a number) in the game data.

Please analyze YOUR bot's performance and improve the strategy based on these results.
"""
    
    if feedback_data.get('validation_errors'):
        feedback_prompt += f"""
CODE VALIDATION ERRORS:
{feedback_data['validation_errors']}

Please fix these validation issues in your code.
"""
    
    feedback_prompt += f"""

IMPROVEMENT INSTRUCTIONS:
1. You are on iteration {iteration} - build upon the previous code above
2. Analyze the errors and game performance data above
3. Fix any syntax, import, or runtime errors from the previous iteration
4. Improve the poker strategy based on game results
5. Ensure the code follows the exact template requirements
6. Return the improved implementation in the same format (Python code block + requirements.txt block)

Focus on making the bot more competitive and error-free. You can see exactly what was wrong with the previous version and should fix those specific issues.
"""
    
    return feedback_prompt


def collect_feedback_data(bot_dir: str, iteration: int) -> dict:
    """Collect feedback data from the verified iteration directory"""
    feedback_data = {
        'success': False,
        'error_message': '',
        'errors': '',
        'game_logs': [],
        'validation_errors': ''
    }
    
    # Use iteration-specific verified directory
    verified_dir = os.path.join(bot_dir, "verified", f"{iteration}_iteration")
    
    # Read current code from iteration-specific directory for potential feedback
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
    if not os.path.exists(verified_dir):
        feedback_data['error_message'] = f"Verified directory for iteration {iteration} not found"
        return feedback_data
    
    # Read error log
    error_log_path = os.path.join(verified_dir, "error.log")
    if os.path.exists(error_log_path):
        with open(error_log_path, 'r', encoding='utf-8') as f:
            error_content = f.read()
            if "No errors detected" in error_content:
                feedback_data['success'] = True
            else:
                feedback_data['errors'] = error_content
                
                # Extract validation errors specifically
                if "Code validation failed" in error_content:
                    lines = error_content.split('\n')
                    validation_lines = [line for line in lines if 'validation' in line.lower() or 'syntax' in line.lower() or 'import' in line.lower()]
                    feedback_data['validation_errors'] = '\n'.join(validation_lines)
    
    # Read all game logs in the iteration directory
    try:
        game_log_files = [f for f in os.listdir(verified_dir) if f.startswith("gamelog_") and f.endswith(".json")]
        print(f"Found {len(game_log_files)} game log files in iteration {iteration}: {game_log_files}")
        
        for log_file in sorted(game_log_files):  # Sort to ensure consistent ordering
            log_path = os.path.join(verified_dir, log_file)
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                    
                    # Extract key performance metrics based on the actual game log structure
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
                    
                    feedback_data['game_logs'].append(game_summary)
                    
            except Exception as e:
                feedback_data['errors'] += f"\nError reading game log {log_file}: {str(e)}"
                
    except Exception as e:
        feedback_data['errors'] += f"\nError listing game log files: {str(e)}"
    
    return feedback_data


def iterative_generation(processor: OpenRouterPromptProcessor, selected_model: dict, original_prompt: str, k_iterations: int, temperature: float, max_tokens: int):
    """Run iterative generation for k iterations with feedback"""
    
    print(f"\n🔄 Starting iterative generation for {k_iterations} iterations...")
    
    current_prompt = original_prompt
    best_result = None
    best_bot_dir = None
    all_results = []  # Track all iterations
    
    # Create a single bot directory for all iterations
    main_bot_dir = None
    
    # Track previous iteration's code for feedback
    previous_code = None
    previous_requirements = None
    
    for iteration in range(1, k_iterations + 1):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {iteration}/{k_iterations}")
        print(f"{'='*60}")
        
        # Process the prompt
        print(f"🚀 Processing prompt with {selected_model['id']}...")
        print("⏳ This may take a moment...")
        
        try:
            result = processor.process_prompt(
                model_id=selected_model['id'],
                prompt=current_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Save result with iteration number (without creating new directories)
            iteration_output_file = f"output_iteration_{iteration}.log"
            processor.save_result_to_log(result, selected_model['id'], current_prompt, iteration_output_file, save_code_blocks=False)
            
            # Extract response content
            if "choices" in result and len(result["choices"]) > 0:
                response_content = result["choices"][0]["message"]["content"]
                python_code, text_content = processor.extract_code_blocks(response_content)
                
                if python_code or text_content:
                    # Store current code for next iteration's feedback
                    previous_code = python_code
                    previous_requirements = text_content
                    
                    if iteration == 1:
                        # Create the main bot directory on first iteration
                        bot_dir = processor.save_code_blocks_to_bot_directory(python_code, text_content, selected_model['id'])
                        main_bot_dir = bot_dir
                    else:
                        # Update the same directory with new code for subsequent iterations
                        bot_dir = processor.save_code_blocks_to_bot_directory(python_code, text_content, selected_model['id'], main_bot_dir)
                    
                    # Save iteration-specific code to iteration directory
                    if bot_dir:
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
                    
                    if bot_dir:
                        print(f"📁 Using bot directory: {bot_dir}")
                        
                        # Test the generated code with iteration number
                        if processor.docker_service:
                            test_success, test_error = processor.test_generated_code(bot_dir, f"{selected_model['id']}_iter_{iteration}", iteration)
                            
                            iteration_result = {
                                'iteration': iteration,
                                'bot_dir': bot_dir,
                                'success': test_success,
                                'error': test_error,
                                'result': result
                            }
                            all_results.append(iteration_result)
                            
                            if test_success:
                                print("✅ Game test completed successfully!")
                                best_result = result
                                best_bot_dir = bot_dir
                                
                                # If successful and this is not the last iteration, create improvement prompt
                                if iteration < k_iterations:
                                    feedback_data = collect_feedback_data(bot_dir, iteration)
                                    feedback_data['success'] = True
                                    current_prompt = create_feedback_prompt(original_prompt, iteration + 1, feedback_data, previous_code, previous_requirements)
                                    print("📈 Creating improvement prompt based on successful game results...")
                                
                            else:
                                print(f"⚠️  Game test failed: {test_error}")
                                
                                # Create feedback prompt for next iteration
                                if iteration < k_iterations:
                                    feedback_data = collect_feedback_data(bot_dir, iteration)
                                    feedback_data['success'] = False
                                    feedback_data['error_message'] = test_error
                                    current_prompt = create_feedback_prompt(original_prompt, iteration + 1, feedback_data, previous_code, previous_requirements)
                                    print("🔧 Creating correction prompt based on errors...")
                        else:
                            print("⚠️  Docker service not available - skipping game test")
                            best_result = result
                            best_bot_dir = bot_dir
                            all_results.append({
                                'iteration': iteration,
                                'bot_dir': bot_dir,
                                'success': True,
                                'error': 'Docker not available',
                                'result': result
                            })
                else:
                    print("⚠️  No code blocks found in response")
                    if iteration < k_iterations:
                        # Create feedback prompt requesting code blocks
                        feedback_data = {
                            'success': False,
                            'error_message': 'No code blocks found in response',
                            'errors': 'The response did not contain the required Python code block and requirements.txt block.'
                        }
                        current_prompt = create_feedback_prompt(original_prompt, iteration + 1, feedback_data, previous_code, previous_requirements)
            
            # Show preview
            if "choices" in result and len(result["choices"]) > 0:
                response_content = result["choices"][0]["message"]["content"]
                print(f"\n📋 Response Preview (first 200 chars):")
                print(f"{response_content[:200]}...")
                
        except Exception as e:
            print(f"❌ Error in iteration {iteration}: {e}")
            all_results.append({
                'iteration': iteration,
                'bot_dir': main_bot_dir,
                'success': False,
                'error': str(e),
                'result': None
            })
            if iteration < k_iterations:
                # Create feedback prompt for the error
                feedback_data = {
                    'success': False,
                    'error_message': str(e),
                    'errors': f'Exception occurred during processing: {str(e)}'
                }
                current_prompt = create_feedback_prompt(original_prompt, iteration + 1, feedback_data, previous_code, previous_requirements)
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 ITERATIVE GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total iterations: {k_iterations}")
    
    if main_bot_dir:
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
        
        # Show latest code location
        latest_iteration = max([int(d.split('_')[0]) for d in os.listdir(verified_dir) if d.endswith('_iteration')]) if os.path.exists(verified_dir) else 0
        if latest_iteration > 0:
            latest_dir = os.path.join(verified_dir, f"{latest_iteration}_iteration")
            print(f"\n🏆 Latest code (iteration {latest_iteration}): {latest_dir}/player.py")
    
    # Show results for each iteration
    for result in all_results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"Iteration {result['iteration']}: {status}")
        if not result['success']:
            print(f"  ⚠️  Error: {result['error']}")
    
    if best_result and best_bot_dir:
        # Find the latest successful iteration
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
    else:
        print("\n⚠️  No successful implementation generated")
    
    return best_result, best_bot_dir


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
            k_iterations = int(input("Number of iterations (default 1): ") or "1")
        except ValueError:
            print("Using default parameters...")
            temperature = 0.7
            max_tokens = 4000
            k_iterations = 1
        
        # Choose between single or iterative generation
        if k_iterations == 1:
            # Single generation (original behavior)
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
        else:
            # Iterative generation
            best_result, best_bot_dir = iterative_generation(
                processor, selected_model, prompt, k_iterations, temperature, max_tokens
            )
        
        print("\n🎉 Processing complete!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 