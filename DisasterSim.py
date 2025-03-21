#!/usr/bin/env python
"""
Script B – Variation of Disaster Magnitude and Dynamics

This script varies shock_magnitude (with fixed disaster_dynamics=2).
Results are written to CSV files, aggregated, and plotted.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
import gc
from DisasterModelNew import DisasterModel

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
    need, incorrect = compute_assistance_metrics(model)
    return (run_id, num_ticks, h_echo, need, incorrect, shock_magnitude, disaster_dynamics)

def run_batch_B(shock_magnitude, disaster_dynamics, num_runs, num_ticks, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["run", "ticks", "h_echo", "need", "incorrect", "shock_magnitude", "disaster_dynamics"]
        writer.writerow(header)
        for run in range(num_runs):
            result = run_single_simulation_B(run, num_ticks, shock_magnitude, disaster_dynamics)
            writer.writerow(result)
            gc.collect()

if __name__ == "__main__":
    num_runs = 10
    num_ticks = 200
    shock_magnitude_values = [1, 2, 4]
    disaster_dynamics_fixed = 2
    filenames = []
    for sm in shock_magnitude_values:
        fname = f"experiment_B_sm_{sm}.csv"
        run_batch_B(sm, disaster_dynamics_fixed, num_runs, num_ticks, fname)
        filenames.append(fname)
    
    def aggregate_experiment(filename):
        data = np.genfromtxt(filename, delimiter=",", skip_header=1)
        return data
    
    results = {}
    for sm in shock_magnitude_values:
        data = aggregate_experiment(f"experiment_B_sm_{sm}.csv")
        results[sm] = {"h_echo": data[:,2], "need": data[:,3], "incorrect": data[:,4]}
    
    def compute_stats(data_list):
        data = np.array(data_list, dtype=float)
        mean = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        return mean, abs(mean-p25), abs(p75-mean)
    
    x = shock_magnitude_values
    h_echo_means = []
    need_means = []
    incorrect_means = []
    h_echo_lower = []
    h_echo_upper = []
    need_lower = []
    need_upper = []
    incorrect_lower = []
    incorrect_upper = []
    for sm in shock_magnitude_values:
        m, l, u = compute_stats(results[sm]["h_echo"])
        h_echo_means.append(m)
        h_echo_lower.append(l)
        h_echo_upper.append(u)
        m2, l2, u2 = compute_stats(results[sm]["need"])
        need_means.append(m2)
        need_lower.append(l2)
        need_upper.append(u2)
        m3, l3, u3 = compute_stats(results[sm]["incorrect"])
        incorrect_means.append(m3)
        incorrect_lower.append(l3)
        incorrect_upper.append(u3)
    
    plt.figure(figsize=(16,4))
    plt.subplot(1,3,1)
    plt.errorbar(x, h_echo_means, yerr=[h_echo_lower, h_echo_upper], fmt='o-', capsize=5, color='blue')
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Human Echo Chamber Metric")
    plt.title("Echo Chamber vs. Shock Magnitude")
    
    plt.subplot(1,3,2)
    plt.errorbar(x, need_means, yerr=[need_lower, need_upper], fmt='o-', capsize=5, color='green')
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Cells in Need Assisted")
    plt.title("Assistance in Need vs. Shock Magnitude")
    
    plt.subplot(1,3,3)
    plt.errorbar(x, incorrect_means, yerr=[incorrect_lower, incorrect_upper], fmt='o-', capsize=5, color='red')
    plt.xlabel("Shock Magnitude")
    plt.ylabel("Incorrect Assistance")
    plt.title("Incorrect Assistance vs. Shock Magnitude")
    
    plt.tight_layout()
    plt.show()
