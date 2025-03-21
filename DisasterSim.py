#!/usr/bin/env python
"""
Script B – Variation of Disaster Magnitude and Dynamics

This script varies the shock_magnitude parameter (with fixed disaster_dynamics=2)
and records the following metrics:
  - Human Echo Chamber Metric
  - AI Echo Chamber Metric
  - Cells in Need Assisted
  - Incorrect Assistance

Each simulation run is executed sequentially (batch processing) and the final
metrics are written to a CSV file. The CSV files are then aggregated and four 
separate error-bar plots are produced (one per metric) versus shock magnitude.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
import gc
from DisasterModelNew import DisasterModel

############################################
# Helper: recursively flatten a nested list.
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
# Standard Metrics Functions
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
        # Gather target cells from pending relief entries (those with 5 elements).
        target_cells = set()
        for entry in agent.pending_relief:
            if len(entry) == 5:
                target_cells.add(entry[4])
        if target_cells and hasattr(agent, "ai_reported") and agent.ai_reported:
            cell_diffs = []
            for cell in target_cells:
                if cell in agent.ai_reported and len(agent.ai_reported[cell]) > 0:
                    rep_values = flatten(agent.ai_reported[cell])
                    try:
                        rep_values = [float(x) for x in rep_values]
                    except Exception:
                        continue
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
# Run a single simulation run for Script B.
############################################
def run_single_simulation_B(run_id, num_ticks, shock_magnitude, disaster_dynamics):
    model = DisasterModel(
        share_exploitative=0.5,
        share_of_disaster=0.2,
        initial_trust=0.5,
        initial_ai_trust=0.75,
        number_of_humans=50,
        share_confirming=0.5,
        disaster_dynamics=disaster_dynamics,
        shock_probability=0.1,
        shock_magnitude=shock_magnitude,
        trust_update_mode="average",
        exploitative_correction_factor=1.0,
        width=50,
        height=50
    )
    for t in range(num_ticks):
        model.step()
    h_echo = compute_echo_chamber_metric(model)
    ai_echo = compute_ai_echo_chamber_metric(model)
    assisted_in_need, assisted_incorrect = compute_assistance_metrics(model)
    del model
    gc.collect()
    return (run_id, num_ticks, h_echo, ai_echo, assisted_in_need, assisted_incorrect, shock_magnitude, disaster_dynamics)

############################################
# Batch Processing Function for Script B.
############################################
def run_batch_B(shock_magnitude, disaster_dynamics, num_runs, num_ticks, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["run", "ticks", "h_echo", "ai_echo", "need", "incorrect", "shock_magnitude", "disaster_dynamics"]
        writer.writerow(header)
        for run in range(num_runs):
            result = run_single_simulation_B(run, num_ticks, shock_magnitude, disaster_dynamics)
            writer.writerow(result)
            gc.collect()

############################################
# Aggregation Function
############################################
def aggregate_experiment(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    return data

def compute_stats(data_list):
    data = np.array(data_list, dtype=float)
    mean = np.mean(data)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    return mean, abs(mean - p25), abs(p75 - mean)

############################################
# Main Execution for Script B.
############################################
if __name__ == "__main__":
    num_runs = 10
    num_ticks = 200
    # Vary shock_magnitude; keep disaster_dynamics fixed.
    shock_magnitude_values = [1, 2, 4]
    disaster_dynamics_fixed = 2
    filenames = []
    for sm in shock_magnitude_values:
        fname = f"experiment_B_sm_{sm}.csv"
        run_batch_B(sm, disaster_dynamics_fixed, num_runs, num_ticks, fname)
        filenames.append(fname)
    
    # Aggregate results by shock_magnitude.
    results = {}
    for sm in shock_magnitude_values:
        data = aggregate_experiment(f"experiment_B_sm_{sm}.csv")
        results[sm] = {
            "h_echo": data[:,2],
            "ai_echo": data[:,3],
            "need": data[:,4],
            "incorrect": data[:,5]
        }
    
    # Compute statistics for each metric.
    x = shock_magnitude_values
    h_echo_means, h_echo_lower, h_echo_upper = [], [], []
    ai_echo_means, ai_echo_lower, ai_echo_upper = [], [], []
    need_means, need_lower, need_upper = [], [], []
    incorrect_means, incorrect_lower, incorrect_upper = [], [], []
    for sm in shock_magnitude_values:
        m, l, u = compute_stats(results[sm]["h_echo"])
        h_echo_means.append(m)
        h_echo_lower.append(l)
        h_echo_upper.append(u)
        m2, l2, u2 = compute_stats(results[sm]["ai_echo"])
        ai_echo_means.append(m2)
        ai_echo_lower.append(l2)
        ai_echo_upper.append(u2)
        m3, l3, u3 = compute_stats(results[sm]["need"])
        need_means.append(m3)
        need_lower.append(l3)
        need_upper.append(u3)
        m4, l4, u4 = compute_stats(results[sm]["incorrect"])
        incorrect_means.append(m4)
        incorrect_lower.append(l4)
        incorrect_upper.append(u4)
    
    # Plot four separate graphs:
    plt.figure(figsize=(16,10))
    
    plt.subplot(2,2,1)
    plt.errorbar(x, h_echo_means, yerr=[h_echo_lower, h_echo_upper], fmt='o-', capsize=5, color='blue')
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Human Echo Chamber Metric")
    plt.title("Human Echo vs. Shock Magnitude")
    
    plt.subplot(2,2,2)
    plt.errorbar(x, ai_echo_means, yerr=[ai_echo_lower, ai_echo_upper], fmt='o-', capsize=5, color='orange')
    plt.xlabel("Shock Magnitude")
    plt.ylabel("AI Echo Chamber Metric")
    plt.title("AI Echo vs. Shock Magnitude")
    
    plt.subplot(2,2,3)
    plt.errorbar(x, need_means, yerr=[need_lower, need_upper], fmt='o-', capsize=5, color='green')
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Cells in Need Assisted")
    plt.title("Assistance in Need vs. Shock Magnitude")
    
    plt.subplot(2,2,4)
    plt.errorbar(x, incorrect_means, yerr=[incorrect_lower, incorrect_upper], fmt='o-', capsize=5, color='red')
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Incorrect Assistance")
    plt.title("Incorrect Assistance vs. Shock Magnitude")
    
    plt.tight_layout()
    plt.show()
