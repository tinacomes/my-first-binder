import numpy as np
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

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

def run_simulation(share_exploitative, num_ticks=300):
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

share_exploitative_values = [0.2, 0.5, 0.8]
num_runs = 5

echo_results = []
assisted_need_results = []
assisted_incorrect_results = []

for se in share_exploitative_values:
    echo_metrics = []
    need_metrics = []
    incorrect_metrics = []
    for run in range(num_runs):
        echo, need, incorrect = run_simulation(se)
        echo_metrics.append(echo)
        need_metrics.append(need)
        incorrect_metrics.append(incorrect)
    echo_results.append(np.mean(echo_metrics))
    assisted_need_results.append(np.mean(need_metrics))
    assisted_incorrect_results.append(np.mean(incorrect_metrics))

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(share_exploitative_values, echo_results, marker='^', linestyle='-')
plt.xlabel("Share Exploitative")
plt.ylabel("Echo Chamber Metric")
plt.title("Echo Chamber vs. Exploring Ratio")

plt.subplot(1,3,2)
plt.plot(share_exploitative_values, assisted_need_results, marker='o', linestyle='-', color='green')
plt.xlabel("Share Exploitative")
plt.ylabel("Cells in Need Assisted")
plt.title("Assistance in Need vs. Exploring Ratio")

plt.subplot(1,3,3)
plt.plot(share_exploitative_values, assisted_incorrect_results, marker='s', linestyle='-', color='red')
plt.xlabel("Share Exploitative")
plt.ylabel("Incorrect Assistance")
plt.title("Incorrect Assistance vs. Exploring Ratio")
plt.tight_layout()
plt.show()

# Rationale:
# By varying the fraction of exploring agents, we test whether a more diverse (broad-search) population yields higher diversity in beliefs
# (i.e. higher echo chamber metric) and how that affects the delivery of assistance.
# A predominance of local, exploitative agents is expected to produce stronger echo chambers.
