#!/usr/bin/env python
"""
Script D – Variation of AI Agent Behaviour (Adaptation vs. No Adaptation)

This script compares simulation runs with AI agents that adapt their output 
to human beliefs (ai_adaptation=True) versus those that do not (ai_adaptation=False).
Results are written to CSV files, aggregated, and a bar chart is plotted.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
import gc
from DisasterModelNew import DisasterModel

# Standard metric functions.
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

# Run a single simulation for Script D.
def run_single_simulation_D(run_id, num_ticks, ai_adaptation):
    # Create the model without passing ai_adaptation (not supported by DisasterModel.__init__).
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
    # If ai_adaptation is False, override each AI agent's provide_information_full method.
    if not ai_adaptation:
        # Define a no-adaptation version that returns raw sensed values.
        def no_adaptation_provide_information_full(self, human_beliefs, trust):
            # Simply return the sensed values without any correction.
            return self.sensed
        # Patch each AI agent.
        for ai in model.ais.values():
            ai.provide_information_full = no_adaptation_provide_information_full.__get__(ai, type(ai))
    
    for t in range(num_ticks):
        model.step()
    h_echo = compute_echo_chamber_metric(model)
    need, incorrect = compute_assistance_metrics(model)
    return (run_id, num_ticks, h_echo, need, incorrect, ai_adaptation)

def run_batch_D(ai_adaptation, num_runs, num_ticks, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["run", "ticks", "h_echo", "need", "incorrect", "ai_adaptation"]
        writer.writerow(header)
        for run in range(num_runs):
            result = run_single_simulation_D(run, num_ticks, ai_adaptation)
            writer.writerow(result)
            gc.collect()

if __name__ == "__main__":
    num_runs = 10
    num_ticks = 200
    output_adapt = "experiment_D_adapt.csv"
    output_noadapt = "experiment_D_noadapt.csv"
    
    run_batch_D(True, num_runs, num_ticks, output_adapt)
    run_batch_D(False, num_runs, num_ticks, output_noadapt)
    
    def aggregate_experiment(filename):
        data = np.genfromtxt(filename, delimiter=",", skip_header=1)
        return data
    
    data_adapt = aggregate_experiment(output_adapt)
    data_noadapt = aggregate_experiment(output_noadapt)
    
    def compute_stats(data_list):
        data = np.array(data_list, dtype=float)
        mean = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        return mean, abs(mean - p25), abs(p75 - mean)
    
    # We compare the human echo chamber metric.
    adapt_vals = data_adapt[:,2]
    noadapt_vals = data_noadapt[:,2]
    mean_adapt, err_low_adapt, err_high_adapt = compute_stats(adapt_vals)
    mean_noadapt, err_low_noadapt, err_high_noadapt = compute_stats(noadapt_vals)
    
    plt.figure(figsize=(8,6))
    plt.bar(["AI Adapts", "AI Does Not Adapt"], [mean_adapt, mean_noadapt],
            yerr=[[err_low_adapt, err_low_noadapt], [err_high_adapt, err_high_noadapt]],
            capsize=5, color=["lightblue", "salmon"])
    plt.ylabel("Human Echo Chamber Metric")
    plt.title("Impact of AI Adaptation on Human Echo Chamber")
    plt.show()
