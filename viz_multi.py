#!/usr/bin/env python3

"""
Multi-Bot Final Iteration Game Results Visualization Script

Reads JSON game logs from multiple bot directories and plots finalDelta for games in the final iteration of each bot.

"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import glob
import re
import sys

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
        print(f"⚠️ Iteration directory not found: {iteration_dir}")
        return game_logs
    json_files = glob.glob(os.path.join(iteration_dir, "gamelog_*.json"))
    json_files.sort()
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
            player_money = game_log.get('playerMoney', {})
            final_delta = player_money.get('finalDelta', {})
            username_mapping = game_log.get('usernameMapping', {})
            bot_player_id = None
            for username, player_id in username_mapping.items():
                if 'test_client' in username.lower() or 'iter' in username.lower():
                    bot_player_id = str(player_id)
                    break
            if bot_player_id and bot_player_id in final_delta:
                final_deltas.append(final_delta[bot_player_id])
                print(f"🎯 Bot finalDelta: {final_delta[bot_player_id]}")
            elif final_delta:
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

def get_final_iteration_dir(bot_dir: str) -> Optional[str]:
    """Find the directory of the final (highest) iteration"""
    if not os.path.exists(bot_dir):
        print(f"❌ Bot directory not found: {bot_dir}")
        return None
    
    verified_dir = os.path.join(bot_dir, "verified")
    if not os.path.exists(verified_dir):
        print(f"❌ Verified directory not found: {verified_dir}")
        return None
    
    iteration_dirs = [os.path.join(verified_dir, item) for item in os.listdir(verified_dir) 
                      if os.path.isdir(os.path.join(verified_dir, item)) and item.endswith('_iteration')]
    if not iteration_dirs:
        print(f"❌ No iteration directories found for {bot_dir}")
        return None
    
    iteration_dirs.sort(key=lambda x: extract_iteration_number(x) or 0)
    return iteration_dirs[-1]  # Last one is the final iteration

def process_bot(bot_dir: str) -> List[float]:
    """Process a single bot and return finalDeltas from its final iteration"""
    final_iter_dir = get_final_iteration_dir(bot_dir)
    if not final_iter_dir:
        return [0.0]  # Default if no data
    
    iteration_num = extract_iteration_number(final_iter_dir)
    print(f"\n📊 Processing final iteration {iteration_num} for bot: {os.path.basename(bot_dir)}")
    
    game_logs = read_game_logs_from_iteration(final_iter_dir)
    if game_logs:
        return extract_bot_final_deltas(game_logs)
    else:
        print(f"📈 No games in final iteration, defaulting to [0.0]")
        return [0.0]

def visualize_multi_bot_final_iterations(bot_dirs: List[str], max_bots: int = 5, save_plot: bool = True, show_plot: bool = True):
    """Create visualization of finalDelta across games in final iteration for multiple bots"""
    bot_dirs = bot_dirs[:max_bots]  # Limit to max_bots
    all_bot_data = {}
    max_games = 0
    
    for bot_dir in bot_dirs:
        bot_name = os.path.basename(bot_dir)
        final_deltas = process_bot(bot_dir)
        all_bot_data[bot_name] = final_deltas
        max_games = max(max_games, len(final_deltas))
    
    if not all_bot_data:
        print("❌ No bot data found")
        return
    
    # Prepare data for plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Multi-Bot Final Iteration Game Results', fontsize=16, fontweight='bold')
    ax.set_title('Final Delta per Game in Final Iteration', fontsize=14)
    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel('Final Delta', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(bot_dirs)))
    game_positions = list(range(1, max_games + 1))
    
    for i, (bot_name, final_deltas) in enumerate(all_bot_data.items()):
        # Pad with zeros if fewer games
        padded_deltas = final_deltas + [0.0] * (max_games - len(final_deltas))
        ax.plot(game_positions, padded_deltas, color=colors[i], linewidth=2, marker='o', markersize=8, label=bot_name)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(game_positions)
    
    plt.tight_layout()
    if save_plot:
        plot_filename = 'multi_bot_final_iteration.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {plot_filename}")
    if show_plot:
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python multi_bot_final_iteration_plot.py <bot_dir1> <bot_dir2> ...")
        print("Example: python multi_bot_final_iteration_plot.py bot/google_gemini_2.0_flash_001_20250719_194006 bot/other_bot")
        return
    bot_dirs = sys.argv[1:]
    visualize_multi_bot_final_iterations(bot_dirs, max_bots=5, save_plot=True, show_plot=True)

if __name__ == "__main__":
    main()