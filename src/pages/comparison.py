import streamlit as st
import json
import os
import re
import matplotlib.pyplot as plt
import sys

# Add root directory to sys.path if needed
sys.path.append(os.getcwd())

def load_winrates(filepath="winrates.json"):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error("Error decoding winrates.json")
        return None

def parse_winrates(data):
    # Data structure to hold plot points: { "pair_key": { "steps": [], "win_rates": [] } }
    plot_data = {}

    # Regex to match key: {step}_{pair_key}_vs_control_{control_step}
    # Example: 600_mewtwoex_vs_mewtwoex_vs_control_500
    pattern = re.compile(r"^(\d+)_(.*)_vs_control_(\d+)$")

    for key, result in data.items():
        match = pattern.match(key)
        if match:
            step = int(match.group(1))
            pair_key = match.group(2)
            # control_step = int(match.group(3)) # Not used for plotting line, maybe for title?

            # Calculate win rate
            total = result.get('total', 0)
            p1_wins = result.get('p1_wins', 0)
            win_rate = (p1_wins / total) if total > 0 else 0.0

            if pair_key not in plot_data:
                plot_data[pair_key] = {'steps': [], 'win_rates': []}

            plot_data[pair_key]['steps'].append(step)
            plot_data[pair_key]['win_rates'].append(win_rate)

    # Sort data points by step
    for pair in plot_data:
        # Use simple list comprehension based sorting to avoid issues
        # Combine steps and win_rates
        combined = sorted(zip(plot_data[pair]['steps'], plot_data[pair]['win_rates']), key=lambda x: x[0])
        plot_data[pair]['steps'] = [x[0] for x in combined]
        plot_data[pair]['win_rates'] = [x[1] for x in combined]

    return plot_data

def main():
    st.set_page_config(page_title="Comparison", layout="wide")

    st.title("Win Rate Comparison")

    # "Share" placeholder - assuming user meant below a section called Share or similar
    # Since I cannot find "Share", I will just place the chart here.

    winrates_data = load_winrates()

    if winrates_data:
        plot_data = parse_winrates(winrates_data)

        if plot_data:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))

            for pair_key, data in plot_data.items():
                ax.plot(data['steps'], data['win_rates'], marker='o', label=pair_key)

            ax.set_xlabel('Checkpoint Step')
            ax.set_ylabel('Win Rate (Target P1 vs Control P2)')
            ax.set_title('Win Rates Over Time')
            ax.legend()
            ax.grid(True)
            ax.set_ylim(0, 1.0)

            st.pyplot(fig)
        else:
            st.warning("No valid data points found in winrates.json.")
    else:
        st.warning("winrates.json not found. Run scripts/plot_winrates.sh to generate data.")

if __name__ == "__main__":
    main()
