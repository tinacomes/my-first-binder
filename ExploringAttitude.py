import numpy as np
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
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

def run_simulation(share_exploitative, num_ticks=100):
    model = DisasterModel(
        share_exploitative=share_exploitative,  # e.g. 0.2 means 20% exploitative, 80% exploratory
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
    for t in range(num_ticks):
        model.step()
    echo_metric = compute_echo_chamber_metric(model)
    assisted_in_need, assisted_incorrect = compute_assistance_metrics(model)
    return echo_metric, assisted_in_need, assisted_incorrect

# Use sorted parameter values for share_exploitative.
share_exploitative_values = [0, 0.1, 0.33, 0.66]
num_runs = 10

# Dictionaries to store lists of metrics for each parameter value.
echo_metrics_by_param = {}
need_metrics_by_param = {}
incorrect_metrics_by_param = {}

for se in share_exploitative_values:
    echo_list = []
    need_list = []
    incorrect_list = []
    for run in range(num_runs):
        echo, need, incorrect = run_simulation(se)
        echo_list.append(echo)
        need_list.append(need)
        incorrect_list.append(incorrect)
    echo_metrics_by_param[se] = echo_list
    need_metrics_by_param[se] = need_list
    incorrect_metrics_by_param[se] = incorrect_list

# Helper function to compute mean, 25th, and 75th percentiles.
def compute_stats(data_list):
    data = np.array(data_list)
    mean = np.mean(data)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    return mean, p25, p75

# Prepare data for errorbar plotting.
x = share_exploitative_values

echo_means = []
echo_err_lower = []
echo_err_upper = []
need_means = []
need_err_lower = []
need_err_upper = []
incorrect_means = []
incorrect_err_lower = []
incorrect_err_upper = []

for se in x:
    mean, p25, p75 = compute_stats(echo_metrics_by_param[se])
    echo_means.append(mean)
    echo_err_lower.append(mean - p25)
    echo_err_upper.append(p75 - mean)
    
    mean, p25, p75 = compute_stats(need_metrics_by_param[se])
    need_means.append(mean)
    need_err_lower.append(mean - p25)
    need_err_upper.append(p75 - mean)
    
    mean, p25, p75 = compute_stats(incorrect_metrics_by_param[se])
    incorrect_means.append(mean)
    incorrect_err_lower.append(mean - p25)
    incorrect_err_upper.append(p75 - mean)

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.errorbar(x, echo_means, yerr=[echo_err_lower, echo_err_upper], fmt='o-', capsize=5)
plt.xlabel("Share Exploitative")
plt.ylabel("Echo Chamber Metric")
plt.title("Echo Chamber vs. Share Exploitative")

plt.subplot(1,3,2)
plt.errorbar(x, need_means, yerr=[need_err_lower, need_err_upper], fmt='o-', capsize=5, color='green')
plt.xlabel("Share Exploitative")
plt.ylabel("Cells in Need Assisted")
plt.title("Assistance in Need vs. Share Exploitative")

plt.subplot(1,3,3)
plt.errorbar(x, incorrect_means, yerr=[incorrect_err_lower, incorrect_err_upper], fmt='o-', capsize=5, color='red')
plt.xlabel("Share Exploitative")
plt.ylabel("Incorrect Assistance")
plt.title("Incorrect Assistance vs. Share Exploitative")

plt.tight_layout()
plt.show()
