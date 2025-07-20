import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import glob
import re
import sys

def extract_iteration_number(dir_path: str) -> Optional[int]:
    match = re.search(r'(\d+)_iteration', dir_path)
    if match:
        return int(match.group(1))
    return None

def read_game_logs_from_iteration(iteration_dir: str) -> List[Dict]:
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

def calculate_iteration_stats(final_deltas: List[float]) -> Tuple[float, float]:
    if not final_deltas:
        return 0.0, 0.0
    mean = np.mean(final_deltas)
    std = np.std(final_deltas) if len(final_deltas) > 1 else 0.0
    return mean, std

def process_bot(bot_dir: str) -> Dict[int, Tuple[float, float]]:
    if not os.path.exists(bot_dir):
        print(f"❌ Bot directory not found: {bot_dir}")
        return {}
    
    verified_dir = os.path.join(bot_dir, "verified")
    if not os.path.exists(verified_dir):
        print(f"❌ Verified directory not found: {verified_dir}")
        return {}
    
    iteration_dirs = [os.path.join(verified_dir, item) for item in os.listdir(verified_dir) 
                      if os.path.isdir(os.path.join(verified_dir, item)) and item.endswith('_iteration')]
    iteration_dirs.sort(key=lambda x: extract_iteration_number(x) or 0)
    
    stats = {}
    for iteration_dir in iteration_dirs:
        iteration_num = extract_iteration_number(iteration_dir)
        if iteration_num is None:
            continue
        game_logs = read_game_logs_from_iteration(iteration_dir)
        final_deltas = extract_bot_final_deltas(game_logs) if game_logs else []
        mean, std = calculate_iteration_stats(final_deltas)
        stats[iteration_num] = (mean, std)
        print(f"📈 Iteration {iteration_num}: Mean {mean:.2f}, Std {std:.2f}")
    return stats

def visualize_multi_bot_stats(bot_dirs: List[str], max_bots: int = 5, save_plot: bool = True, show_plot: bool = True):
    bot_dirs = bot_dirs[:max_bots]
    all_bot_data = {}
    all_iterations = set()
    
    for bot_dir in bot_dirs:
        bot_name = os.path.basename(bot_dir)
        print(f"\n📊 Processing bot: {bot_name}")
        stats = process_bot(bot_dir)
        all_bot_data[bot_name] = stats
        all_iterations.update(stats.keys())
    
    if not all_bot_data:
        print("❌ No bot data found")
        return
    
    sorted_iterations = sorted(all_iterations)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Multi-Bot Average Final Delta with Variance Bands', fontsize=16, fontweight='bold')
    ax.set_title('Average Final Delta per Iteration (Shaded: ± Std Dev)', fontsize=14)
    ax.set_xlabel('Iteration Number', fontsize=12)
    ax.set_ylabel('Average Final Delta', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(bot_dirs)))
    for i, (bot_name, stats) in enumerate(all_bot_data.items()):
        x = sorted_iterations
        means = [stats.get(iter_num, (0.0, 0.0))[0] for iter_num in x]
        stds = [stats.get(iter_num, (0.0, 0.0))[1] for iter_num in x]
        
        ax.plot(x, means, color=colors[i], linewidth=2, marker='o', markersize=8, label=bot_name)
        ax.fill_between(x, np.array(means) - np.array(stds), np.array(means) + np.array(stds), 
                        color=colors[i], alpha=0.2)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticks(sorted_iterations)
    
    plt.tight_layout()
    if save_plot:
        plot_filename = 'multi_bot_avg_variance.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {plot_filename}")
    if show_plot:
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python multi_bot_avg_variance_plot.py <bot_dir1> <bot_dir2> ...")
        print("Example: python multi_bot_avg_variance_plot.py bot/google_gemini_2.0_flash_001_20250719_194006 bot/other_bot")
        return
    bot_dirs = sys.argv[1:]
    visualize_multi_bot_stats(bot_dirs, max_bots=5, save_plot=True, show_plot=True)

if __name__ == "__main__":
    main()