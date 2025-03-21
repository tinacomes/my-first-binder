#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
import gc
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
# Import your updated DisasterModel and HumanAgent from your module.
from DisasterModelNew import DisasterModel, HumanAgent

############################################
# Helper function to recursively flatten a nested list.
############################################
def flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat

############################################
# Standard Metrics
############################################
def compute_echo_chamber_metric(model):
    differences = []
    for agent in model.humans.values():
        if agent.friends:
            my_avg = np.mean(list(agent.beliefs.values()))
            friend_avgs = []
            for fid in agent.friends:
                if fid in model.humans:
                    friend_avgs.append(np.mean(list(model.humans[fid].beliefs.values())))
            if friend_avgs:
                differences.append(abs(my_avg - np.mean(friend_avgs)))
    return np.mean(differences) if differences else None

def compute_ai_echo_chamber_metric(model):
    differences = []
    for agent in model.humans.values():
        target_cells = set()
        for entry in agent.pending_relief:
            if len(entry) == 5:
                target_cells.add(entry[4])
        if target_cells and hasattr(agent, "ai_reported") and agent.ai_reported:
            cell_diffs = []
            for cell in target_cells:
                if cell in agent.ai_reported and len(agent.ai_reported[cell]) > 0:
                    rep_values = flatten(agent.ai_reported[cell])
                    rep_values = [float(x) for x in rep_values]
                    if rep_values:
                        avg_ai = np.mean(rep_values)
                        diff = abs(agent.beliefs[cell] - avg_ai)
                        cell_diffs.append(diff)
            if cell_diffs:
                differences.append(np.mean(cell_diffs))
    return np.mean(differences) if differences else None

def compute_assistance_metrics(model):
    assisted_in_need = 0
    assisted_incorrect = 0
    for pos, level in model.disaster_grid.items():
        if level >= 4:
            tokens = model.assistance_exploit.get(pos, 0) + model.assistance_explor.get(pos, 0)
            if tokens > 0:
                assisted_in_need += 1
        if level <= 2:
            tokens_incorrect = model.assistance_incorrect_exploit.get(pos, 0) + model.assistance_incorrect_explor.get(pos, 0)
            if tokens_incorrect > 0:
                assisted_incorrect += 1
    return assisted_in_need, assisted_incorrect

############################################
# Run a single simulation run and record time-series data.
############################################
def run_model_collect_timeseries(num_ticks=300):
    model = DisasterModel(
        share_exploitative=0.01,
        share_of_disaster=0.2,
        initial_trust=0.5,
        initial_ai_trust=0.75,
        number_of_humans=50,
        share_confirming=0.5,
        disaster_dynamics=2,
        shock_probability=0.1,
        shock_magnitude=2,
        trust_update_mode="average",
        exploitative_correction_factor=1.0,
        width=50,
        height=50
    )
    ticks = []
    human_echo_ts = []
    ai_echo_ts = []
    trust_ai_ts = []   # average AI trust across agents
    unmet_ts = []
    assist_need_ts = []
    assist_incorrect_ts = []
    for t in range(num_ticks):
        model.step()
        ticks.append(t)
        h_echo = compute_echo_chamber_metric(model)
        a_echo = compute_ai_echo_chamber_metric(model)
        human_echo_ts.append(h_echo)
        ai_echo_ts.append(a_echo)
        # Compute average AI trust
        ai_trust_list = []
        for agent in model.humans.values():
            ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
            if ai_vals:
                ai_trust_list.append(np.mean(ai_vals))
        trust_ai_ts.append(np.mean(ai_trust_list) if ai_trust_list else 0)
        unmet = 0
        for pos, level in model.disaster_grid.items():
            if level >= 4 and model.tokens_this_tick.get(pos, 0) == 0:
                unmet += 1
        unmet_ts.append(unmet)
        need, incorrect = compute_assistance_metrics(model)
        assist_need_ts.append(need)
        assist_incorrect_ts.append(incorrect)
    return ticks, human_echo_ts, ai_echo_ts, trust_ai_ts, unmet_ts, assist_need_ts, assist_incorrect_ts

############################################
# Batch runs: Write each run's timeseries data to a CSV file.
############################################
def run_batch(num_runs, num_ticks, output_filename):
    header = ["run", "tick", "human_echo", "ai_echo", "trust_ai", "unmet", "assist_need", "assist_incorrect"]
    with open(output_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for run in range(num_runs):
            print(f"Starting run {run+1}/{num_runs}")
            ticks, human_echo_ts, ai_echo_ts, trust_ai_ts, unmet_ts, assist_need_ts, assist_incorrect_ts = run_model_collect_timeseries(num_ticks)
            for i in range(num_ticks):
                row = [run, ticks[i], human_echo_ts[i], ai_echo_ts[i], trust_ai_ts[i], unmet_ts[i], assist_need_ts[i], assist_incorrect_ts[i]]
                writer.writerow(row)
            # Clear model and run garbage collection.
            gc.collect()
            print(f"Completed run {run+1}/{num_runs}")

############################################
# Aggregation: Read CSV data and aggregate time-series by tick.
############################################
def aggregate_timeseries_from_csv(filename, num_runs, num_ticks):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    # Data columns: run, tick, human_echo, ai_echo, trust_ai, unmet, assist_need, assist_incorrect
    # Reshape into (num_runs, num_ticks, num_columns)
    reshaped = data.reshape((num_runs, num_ticks, -1))
    # We'll aggregate columns 2 through 7 (human_echo, ai_echo, trust_ai, unmet, assist_need, assist_incorrect)
    aggregated = {}
    col_names = ["human_echo", "ai_echo", "trust_ai", "unmet", "assist_need", "assist_incorrect"]
    for i, col in enumerate(col_names, start=2):
        col_data = reshaped[:, :, i]
        mean = np.mean(col_data, axis=0)
        p25 = np.percentile(col_data, 25, axis=0)
        p75 = np.percentile(col_data, 75, axis=0)
        aggregated[col] = (mean, p25, p75)
    return reshaped[0, :, 1], aggregated  # ticks and aggregated metrics

############################################
# Main execution
############################################
if __name__ == "__main__":
    num_runs = 10
    num_ticks = 300
    output_filename = "timeseries_data.csv"
    
    # Run simulations in batch and write to file.
    run_batch(num_runs, num_ticks, output_filename)
    
    # Aggregate results.
    ticks, aggregated = aggregate_timeseries_from_csv(output_filename, num_runs, num_ticks)
    
    # Plot each metric over time with error bands.
    plt.figure(figsize=(16,12))
    metrics = aggregated.keys()
    plot_positions = {"human_echo": 1, "ai_echo": 2, "trust_ai": 3, "unmet": 4, "assist_need": 5, "assist_incorrect": 6}
    titles = {
        "human_echo": "Human Echo Chamber Metric",
        "ai_echo": "AI Echo Chamber Metric",
        "trust_ai": "Average Trust in AI",
        "unmet": "Unmet Needs (Cells)",
        "assist_need": "Cells in Need Assisted",
        "assist_incorrect": "Incorrect Assistance"
    }
    for key in metrics:
        mean, p25, p75 = aggregated[key]
        pos = plot_positions[key]
        plt.subplot(3,2,pos)
        plt.plot(ticks, mean, label=f"{key} mean")
        plt.fill_between(ticks, mean - (mean - p25), mean + (p75 - mean), alpha=0.3, label=f"{key} 25th-75th")
        plt.xlabel("Tick")
        plt.ylabel(key)
        plt.title(titles[key])
        plt.legend()
    plt.tight_layout()
    plt.show()
