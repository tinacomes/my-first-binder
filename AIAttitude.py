#!/usr/bin/env python
"""
Script D – Variation of AI Agent Behaviour (Adaptation vs. No Adaptation)

This script compares simulation runs with AI agents that adapt their output 
to human beliefs versus those that do not.
It outputs four metrics:
  - Human Echo Chamber Metric
  - AI Echo Chamber Metric
  - Cells in Need Assisted
  - Incorrect Assistance

Results are written to CSV files (one for each condition), then aggregated and 
plotted as two separate graphs:
  (a) Echo Chamber Effects (Human and AI Echo)
  (b) Assistance Metrics (Cells in Need Assisted and Incorrect Assistance)
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
# Run a single simulation run for Script D.
############################################
def run_single_simulation_D(run_id, num_ticks, ai_adaptation):
    # Create model (do not pass ai_adaptation; instead, override AI behavior after creation).
    model = DisasterModel(
        share_exploitative=0.5,
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
    # If AI adaptation is False, override each AI agent's provide_information_full method.
    if not ai_adaptation:
        def no_adaptation_provide_information_full(self, human_beliefs, trust):
            # Return raw sensed values.
            return self.sensed
        for ai in model.ais.values():
            ai.provide_information_full = no_adaptation_provide_information_full.__get__(ai, type(ai))
    for t in range(num_ticks):
        model.step()
    h_echo = compute_echo_chamber_metric(model)
    ai_echo = compute_ai_echo_chamber_metric(model)
    assisted_in_need, assisted_incorrect = compute_assistance_metrics(model)
    del model
    gc.collect()
    return (run_id, num_ticks, h_echo, ai_echo, assisted_in_need, assisted_incorrect, ai_adaptation)

############################################
# Batch Processing for Script D.
############################################
def run_batch_D(ai_adaptation, num_runs, num_ticks, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["run", "ticks", "h_echo", "ai_echo", "need", "incorrect", "ai_adaptation"]
        writer.writerow(header)
        for run in range(num_runs):
            result = run_single_simulation_D(run, num_ticks, ai_adaptation)
            writer.writerow(result)
            gc.collect()

############################################
# Aggregation and Plotting for Script D.
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

if __name__ == "__main__":
    num_runs = 10
    num_ticks = 200
    output_adapt = "experiment_D_adapt.csv"
    output_noadapt = "experiment_D_noadapt.csv"
    
    run_batch_D(True, num_runs, num_ticks, output_adapt)
    run_batch_D(False, num_runs, num_ticks, output_noadapt)
    
    data_adapt = aggregate_experiment(output_adapt)
    data_noadapt = aggregate_experiment(output_noadapt)
    
    # Data columns: run, ticks, h_echo, ai_echo, need, incorrect, ai_adaptation
    def compute_stats_for_metric(data):
        return compute_stats(data)
    
    # Extract metrics.
    adapt_h_echo, adapt_ai_echo = data_adapt[:,2], data_adapt[:,3]
    adapt_need, adapt_incorrect = data_adapt[:,4], data_adapt[:,5]
    
    noadapt_h_echo, noadapt_ai_echo = data_noadapt[:,2], data_noadapt[:,3]
    noadapt_need, noadapt_incorrect = data_noadapt[:,4], data_noadapt[:,5]
    
    # Compute stats for each metric.
    adapt_stats = {
        "h_echo": compute_stats_for_metric(adapt_h_echo),
        "ai_echo": compute_stats_for_metric(adapt_ai_echo),
        "need": compute_stats_for_metric(adapt_need),
        "incorrect": compute_stats_for_metric(adapt_incorrect)
    }
    noadapt_stats = {
        "h_echo": compute_stats_for_metric(noadapt_h_echo),
        "ai_echo": compute_stats_for_metric(noadapt_ai_echo),
        "need": compute_stats_for_metric(noadapt_need),
        "incorrect": compute_stats_for_metric(noadapt_incorrect)
    }
    
    ############################################
    # Graph (a): Echo Chamber Effects
    ############################################
    labels = ["Human Echo", "AI Echo"]
    adapt_means_a = [adapt_stats["h_echo"][0], adapt_stats["ai_echo"][0]]
    noadapt_means_a = [noadapt_stats["h_echo"][0], noadapt_stats["ai_echo"][0]]
    adapt_err_a = [[adapt_stats["h_echo"][1], adapt_stats["ai_echo"][1]],
                   [adapt_stats["h_echo"][2], adapt_stats["ai_echo"][2]]]
    noadapt_err_a = [[noadapt_stats["h_echo"][1], noadapt_stats["ai_echo"][1]],
                     [noadapt_stats["h_echo"][2], noadapt_stats["ai_echo"][2]]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, adapt_means_a, width, yerr=adapt_err_a, capsize=5, color="lightblue", label="AI Adapts")
    plt.bar(x + width/2, noadapt_means_a, width, yerr=noadapt_err_a, capsize=5, color="salmon", label="AI Does Not Adapt")
    plt.xticks(x, labels)
    plt.ylabel("Metric Value")
    plt.title("Echo Chamber Effects")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    ############################################
    # Graph (b): Assistance Metrics
    ############################################
    labels2 = ["Cells in Need", "Incorrect Assistance"]
    adapt_means_b = [adapt_stats["need"][0], adapt_stats["incorrect"][0]]
    noadapt_means_b = [noadapt_stats["need"][0], noadapt_stats["incorrect"][0]]
    adapt_err_b = [[adapt_stats["need"][1], adapt_stats["incorrect"][1]],
                   [adapt_stats["need"][2], adapt_stats["incorrect"][2]]]
    noadapt_err_b = [[noadapt_stats["need"][1], noadapt_stats["incorrect"][1]],
                     [noadapt_stats["need"][2], noadapt_stats["incorrect"][2]]]
    
    x2 = np.arange(len(labels2))
    width2 = 0.35
    
    plt.figure(figsize=(10,6))
    plt.bar(x2 - width2/2, adapt_means_b, width2, yerr=adapt_err_b, capsize=5, color="lightblue", label="AI Adapts")
    plt.bar(x2 + width2/2, noadapt_means_b, width2, yerr=noadapt_err_b, capsize=5, color="salmon", label="AI Does Not Adapt")
    plt.xticks(x2, labels2)
    plt.ylabel("Metric Value")
    plt.title("Assistance Metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()
