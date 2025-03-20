import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from DisasterModelNew import DisasterModel

# (Assume here the full definitions of DisasterModel, HumanAgent, and AIAgent are available, 
# with our modifications from previous scripts.)

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

def run_simulation(share_confirming, num_ticks=300):
    model = DisasterModel(
        share_exploitative=0.5,
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
    echo_metric = compute_echo_chamber_metric(model)
    assisted_in_need, assisted_incorrect = compute_assistance_metrics(model)
    return echo_metric, assisted_in_need, assisted_incorrect

share_confirming_values = [0.2, 0.5, 0.8]
num_runs = 10

# Store metrics for each parameter value.
echo_data = {}
need_data = {}
incorrect_data = {}

for sc in share_confirming_values:
    echo_list = []
    need_list = []
    incorrect_list = []
    for run in range(num_runs):
        echo, need, incorrect = run_simulation(sc)
        echo_list.append(echo)
        need_list.append(need)
        incorrect_list.append(incorrect)
    echo_data[sc] = echo_list
    need_data[sc] = need_list
    incorrect_data[sc] = incorrect_list

# Plotting with percentiles.
def plot_with_errorbars(x_vals, data_dict, ylabel, title):
    means = []
    lower = []  # difference from 25th percentile to mean
    upper = []  # difference from mean to 75th percentile
    for x in x_vals:
        data = np.array(data_dict[x])
        mean = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        means.append(mean)
        lower.append(mean - p25)
        upper.append(p75 - mean)
    plt.errorbar(x_vals, means, yerr=[lower, upper], fmt='o-', capsize=5)
    plt.xlabel("Parameter Value")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plot_with_errorbars(share_confirming_values, echo_data, "Echo Chamber Metric", "Echo Chamber vs. Share Confirming")
plt.subplot(1,3,2)
plot_with_errorbars(share_confirming_values, need_data, "Cells in Need Assisted", "Assistance in Need vs. Share Confirming")
plt.subplot(1,3,3)
plot_with_errorbars(share_confirming_values, incorrect_data, "Cells Incorrectly Assisted", "Incorrect Assistance vs. Share Confirming")
plt.tight_layout()
plt.show()

# Rationale:
# By varying the share of confirming agents, we test whether predisposition to confirm reinforces echo chambers 
# (i.e. lower divergence in opinions among friends) and whether it changes the pattern of assistance.
# Displaying percentiles (the 25th, 50th, and 75th) along with the mean offers insight into the variability of outcomes.
