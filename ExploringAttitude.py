#!/usr/bin/env python
"""
Optimized Batch Simulation Script

This script runs each simulation run in a separate process (one at a time)
and writes the final output metrics (e.g., echo chamber metrics and assistance metrics)
to a CSV file immediately. This minimizes memory usage.

It assumes that your updated DisasterModelNew (including HumanAgent and AIAgent)
is available for import.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
import gc
import multiprocessing as mp
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

# Import your model components (adjust the import path as needed)
from DisasterModelNew import DisasterModel, HumanAgent

############################################
# Helper function: recursively flatten a nested list.
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
# Single Simulation Run Function
############################################
def run_single_simulation(params):
    """
    Run a single simulation with given parameters.
    params: tuple (run_id, num_ticks, share_exploitative, share_confirming)
    Returns a tuple with:
       run_id, num_ticks, human_echo, ai_echo, assisted_in_need, assisted_incorrect,
       share_exploitative, share_confirming.
    """
    run_id, num_ticks, share_exploitative, share_confirming = params
    model = DisasterModel(
        share_exploitative=share_exploitative,
        share_of_disaster=0.2,
        initial_trust=0.5,
        initial_ai_trust=0.75,
        number_of_humans=50,
        share_confirming=share_confirming,
        disaster_dynamics=2,
        shock_probability=0.1,
        shock_magnitude=2,
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
    # Clean up model to free memory.
    del model
    gc.collect()
    return (run_id, num_ticks, h_echo, ai_echo, assisted_in_need, assisted_incorrect, share_exploitative, share_confirming)

############################################
# Batch Processing Function
############################################
def run_batch_experiments(num_runs, num_ticks, param_values, param_name, fixed_params):
    """
    param_values: list of values for the parameter to vary.
    param_name: string name, e.g., "share_exploitative" or "share_confirming"
    fixed_params: dictionary of fixed parameters for the simulation.
    Writes one CSV file per param value.
    """
    for val in param_values:
        output_filename = f"experiment_{param_name}_{val}.csv"
        with open(output_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["run", "tick", "h_echo", "ai_echo", "need", "incorrect", param_name]
            writer.writerow(header)
            params = []
            for run in range(num_runs):
                # Combine fixed parameters and the varying parameter.
                if param_name == "share_exploitative":
                    sp = val
                    sc = fixed_params.get("share_confirming", 0.5)
                else:
                    sp = fixed_params.get("share_exploitative", 0.5)
                    sc = val
                params.append((run, num_ticks, sp, sc))
            # Use a pool with one process to run sequentially.
            with mp.Pool(processes=1) as pool:
                for res in pool.imap_unordered(run_single_simulation, params):
                    writer.writerow(res)
                    gc.collect()
        print(f"Completed batch for {param_name} = {val}")

############################################
# Aggregation Function
############################################
def aggregate_experiment(filenames, param_name):
    all_data = []
    for fname in filenames:
        data = np.genfromtxt(fname, delimiter=",", skip_header=1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    # Group data by the parameter.
    groups = {}
    for row in all_data:
        key = row[-1]
        groups.setdefault(key, []).append(row)
    # Convert each group's list to numpy array.
    for key in groups:
        groups[key] = np.array(groups[key])
    return groups

def compute_stats(data_list):
    data = np.array(data_list, dtype=float)
    mean = np.mean(data)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    return mean, abs(mean - p25), abs(p75 - mean)

def prepare_plot_data(grouped, key_index):
    # key_index: column index for metric (2: h_echo, 3: ai_echo, 4: need, 5: incorrect)
    means = []
    lowers = []
    uppers = []
    param_keys = sorted(grouped.keys())
    for k in param_keys:
        col_data = grouped[k][:, key_index]
        mean, err_low, err_high = compute_stats(col_data)
        means.append(mean)
        lowers.append(err_low)
        uppers.append(err_high)
    return param_keys, means, lowers, uppers

############################################
# Main Execution for Scripts A-D
############################################
if __name__ == "__main__":
    num_runs = 10
    num_ticks = 200
    
    # Script A: Vary share_confirming while keeping share_exploitative fixed.
    share_confirming_values = [0.2, 0.5, 0.8]
    fixed_params_A = {"share_exploitative": 0.5}
    filenames_A = []
    for sc in share_confirming_values:
        fname = f"experiment_share_confirming_{sc}.csv"
        run_batch_experiments(num_runs, num_ticks, [sc], "share_confirming", fixed_params_A)
        filenames_A.append(fname)
    
    # Script B: Vary shock_magnitude (for example) with fixed disaster_dynamics.
    # (Omitted here for brevity; similar structure as above.)
    
    # Script C: Vary share_exploitative while keeping share_confirming fixed.
    share_exploitative_values = [0.2, 0.5, 0.8]
    fixed_params_C = {"share_confirming": 0.5}
    filenames_C = []
    for sp in share_exploitative_values:
        fname = f"experiment_share_exploitative_{sp}.csv"
        run_batch_experiments(num_runs, num_ticks, [sp], "share_exploitative", fixed_params_C)
        filenames_C.append(fname)
    
    # Script D: Vary AI adaptation (if supported by your model).
    # (Omitted here for brevity.)
    
    # Aggregation and plotting for Script A (as an example)
    grouped_A = aggregate_experiment(filenames_A, "share_confirming")
    param_keys_A, h_echo_means_A, h_echo_lower_A, h_echo_upper_A = prepare_plot_data(grouped_A, 2)
    param_keys_A, ai_echo_means_A, ai_echo_lower_A, ai_echo_upper_A = prepare_plot_data(grouped_A, 3)
    param_keys_A, need_means_A, need_lower_A, need_upper_A = prepare_plot_data(grouped_A, 4)
    param_keys_A, incorrect_means_A, incorrect_lower_A, incorrect_upper_A = prepare_plot_data(grouped_A, 5)
    
    plt.figure(figsize=(16,4))
    plt.subplot(1,4,1)
    plt.errorbar(param_keys_A, h_echo_means_A, yerr=[h_echo_lower_A, h_echo_upper_A], fmt='o-', capsize=5)
    plt.xlabel("Share Confirming")
    plt.ylabel("Human Echo Chamber Metric")
    plt.title("Human Echo vs. Share Confirming")
    
    plt.subplot(1,4,2)
    plt.errorbar(param_keys_A, ai_echo_means_A, yerr=[ai_echo_lower_A, ai_echo_upper_A], fmt='o-', capsize=5, color='orange')
    plt.xlabel("Share Confirming")
    plt.ylabel("AI Echo Chamber Metric")
    plt.title("AI Echo vs. Share Confirming")
    
    plt.subplot(1,4,3)
    plt.errorbar(param_keys_A, need_means_A, yerr=[need_lower_A, need_upper_A], fmt='o-', capsize=5, color='green')
    plt.xlabel("Share Confirming")
    plt.ylabel("Cells in Need Assisted")
    plt.title("Assistance in Need vs. Share Confirming")
    
    plt.subplot(1,4,4)
    plt.errorbar(param_keys_A, incorrect_means_A, yerr=[incorrect_lower_A, incorrect_upper_A], fmt='o-', capsize=5, color='red')
    plt.xlabel("Share Confirming")
    plt.ylabel("Incorrect Assistance")
    plt.title("Incorrect Assistance vs. Share Confirming")
    
    plt.tight_layout()
    plt.show()
