#!/usr/bin/env python3
"""
Performance Visualization Script
Reads JSON game logs from main bot directory and plots finalDelta for each game across iterations.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import glob
import re

def extract_iteration_number(dir_path: str) -> Optional[int]:
    """Extract iteration number from directory path"""
    match = re.search(r'(\d+)_iteration', dir_path)
    if match:
        return int(match.group(1))
    return None

def read_game_logs_from_iteration(iteration_dir: str) -> List[Dict]:
    """Read all JSON game log files from an iteration directory"""
    game_logs = []
    
    if not os.path.exists(iteration_dir):
        print(f"⚠️  Iteration directory not found: {iteration_dir}")
        return game_logs
    
    # Find all JSON files in the iteration directory
    json_files = glob.glob(os.path.join(iteration_dir, "gamelog_*.json"))
    
    # Sort by numerical game ID instead of alphabetically
    def extract_game_id(filename):
        match = re.search(r'gamelog_(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        return 0
    
    json_files.sort(key=extract_game_id)  # Sort by numerical game ID
    
    print(f"📁 Found {len(json_files)} JSON files in {iteration_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
                game_logs.append(game_data)
                print(f"✅ Loaded: {os.path.basename(json_file)}")
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")
    
    return game_logs

def extract_bot_final_deltas(game_logs: List[Dict]) -> List[float]:
    """Extract finalDelta values for the bot from game logs"""
    final_deltas = []
    
    for game_log in game_logs:
        try:
            # Get player money data
            player_money = game_log.get('playerMoney', {})
            final_delta = player_money.get('finalDelta', {})
            
            # Find bot player ID (usually the one with 'test_client' or 'iter' in username)
            username_mapping = game_log.get('usernameMapping', {})
            bot_player_id = None
            
            for username, player_id in username_mapping.items():
                if 'test_client' in username.lower() or 'iter' in username.lower():
                    bot_player_id = str(player_id)
                    break
            
            if bot_player_id and bot_player_id in final_delta:
                final_deltas.append(final_delta[bot_player_id])
                print(f"🎯 Bot finalDelta: {final_delta[bot_player_id]}")
            else:
                # If bot not found, use the first available player
                if final_delta:
                    first_delta = list(final_delta.values())[0]
                    final_deltas.append(first_delta)
                    print(f"🎯 Using first player finalDelta: {first_delta}")
                else:
                    final_deltas.append(0.0)
                    print(f"🎯 No finalDelta found, defaulting to 0")
                    
        except Exception as e:
            print(f"❌ Error extracting finalDelta: {e}")
            final_deltas.append(0.0)
    
    return final_deltas

def visualize_performance(main_bot_dir: str, save_plot: bool = True, show_plot: bool = True):
    """
    Create visualization of finalDelta performance across iterations
    
    Args:
        main_bot_dir: Path to the main bot directory
        save_plot: Whether to save the plot as PNG
        show_plot: Whether to display the plot
    """
    
    if not os.path.exists(main_bot_dir):
        print(f"❌ Main bot directory not found: {main_bot_dir}")
        return
    
    print(f"📊 Creating performance visualization for: {main_bot_dir}")
    
    # Find verified directory
    verified_dir = os.path.join(main_bot_dir, "verified")
    if not os.path.exists(verified_dir):
        print(f"❌ Verified directory not found: {verified_dir}")
        return
    
    # Find all iteration directories
    iteration_dirs = []
    for item in os.listdir(verified_dir):
        item_path = os.path.join(verified_dir, item)
        if os.path.isdir(item_path) and item.endswith('_iteration'):
            iteration_dirs.append(item_path)
    
    iteration_dirs.sort(key=lambda x: extract_iteration_number(x) or 0)
    
    print(f"🔄 Found {len(iteration_dirs)} iteration directories")
    
    # Collect data for each iteration
    all_iterations = []
    all_final_deltas = []
    max_games = 0
    
    for iteration_dir in iteration_dirs:
        iteration_num = extract_iteration_number(iteration_dir)
        if iteration_num is None:
            continue
            
        print(f"\n📁 Processing iteration {iteration_num}: {iteration_dir}")
        
        # Read game logs for this iteration
        game_logs = read_game_logs_from_iteration(iteration_dir)
        
        if game_logs:
            # Extract finalDelta values
            final_deltas = extract_bot_final_deltas(game_logs)
            all_iterations.append(iteration_num)
            all_final_deltas.append(final_deltas)
            max_games = max(max_games, len(final_deltas))
            print(f"📈 Iteration {iteration_num}: {len(final_deltas)} games, deltas: {final_deltas}")
        else:
            # No JSON files found, default to 0
            all_iterations.append(iteration_num)
            all_final_deltas.append([0.0])
            print(f"📈 Iteration {iteration_num}: No games, defaulting to [0.0]")
    
    if not all_iterations:
        print("❌ No iteration data found")
        return
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Poker Bot Performance Analysis\n{os.path.basename(main_bot_dir)}', fontsize=16, fontweight='bold')
    
    # Plot: Individual game finalDelta values
    ax.set_title('Final Delta by Game and Iteration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel('Final Delta', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Create x-axis positions for games
    game_positions = list(range(1, max_games + 1))
    
    # Plot each iteration
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_iterations)))
    
    for i, (iteration_num, final_deltas) in enumerate(zip(all_iterations, all_final_deltas)):
        # Pad with zeros if this iteration has fewer games
        padded_deltas = final_deltas + [0.0] * (max_games - len(final_deltas))
        
        # Plot individual points
        ax.scatter(game_positions, padded_deltas, 
                   c=[colors[i]], s=100, alpha=0.7, 
                   label=f'Iteration {iteration_num}', edgecolors='black', linewidth=1)
        
        # Connect points with lines
        ax.plot(game_positions, padded_deltas, 
                c=colors[i], alpha=0.5, linewidth=2)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(game_positions)
    
    # Add summary statistics
    total_games = sum(len(deltas) for deltas in all_final_deltas)
    overall_avg = sum(sum(deltas) for deltas in all_final_deltas) / total_games if total_games > 0 else 0
    
    stats_text = f'Summary:\nTotal Games: {total_games}\nOverall Average: {overall_avg:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        plot_filename = os.path.join(main_bot_dir, 'performance_visualization.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {plot_filename}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    # Print summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"=" * 50)
    print(f"Total Iterations: {len(all_iterations)}")
    print(f"Total Games: {total_games}")
    print(f"Overall Average Final Delta: {overall_avg:.2f}")
    print(f"\nPer-Iteration Averages:")
    for iteration_num, final_deltas in zip(all_iterations, all_final_deltas):
        if final_deltas:
            avg = sum(final_deltas) / len(final_deltas)
        else:
            avg = 0.0
        print(f"  Iteration {iteration_num}: {avg:.2f}")
    
    return fig, ax

def main():
    """Main function to run the visualization"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_performance.py <main_bot_dir>")
        print("Example: python visualize_performance.py bot/google_gemini_2.0_flash_001_20250719_194006")
        return
    
    main_bot_dir = sys.argv[1]
    
    # Check if directory exists
    if not os.path.exists(main_bot_dir):
        print(f"❌ Directory not found: {main_bot_dir}")
        return
    
    # Create visualization
    try:
        visualize_performance(main_bot_dir, save_plot=True, show_plot=True)
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 