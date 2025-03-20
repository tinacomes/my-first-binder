import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
import networkx as nx
from DisasterModelNew import DisasterModel

# (Re-use the DisasterModel, HumanAgent, AIAgent definitions from above.)
# For brevity, assume that the full base code (with our modifications) is defined here.
# We then add a helper function to compute the echo chamber metric.

def compute_echo_chamber_metric(model):
    """
    For each human agent, compute the absolute difference between
    its average belief (across all grid cells) and the average belief of its friends.
    Then, average these differences over all agents.
    """
    differences = []
    for agent in model.humans.values():
        if agent.friends:
            my_avg = np.mean(list(agent.beliefs.values()))
            friend_avgs = []
            for fid in agent.friends:
                if fid in model.humans:
                    friend_agent = model.humans[fid]
                    friend_avgs.append(np.mean(list(friend_agent.beliefs.values())))
            if friend_avgs:
                differences.append(abs(my_avg - np.mean(friend_avgs)))
    return np.mean(differences) if differences else None

def run_simulation(share_confirming, num_ticks=300):
    # Use fixed parameters except for share_confirming.
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
    return compute_echo_chamber_metric(model)

# Parameter sweep for share_confirming
share_confirming_values = [0.2, 0.5, 0.8]
num_runs = 5
results = []

for sc in share_confirming_values:
    run_metrics = []
    for run in range(num_runs):
        metric = run_simulation(sc)
        run_metrics.append(metric)
    results.append(np.mean(run_metrics))

plt.figure()
plt.plot(share_confirming_values, results, marker='o', linestyle='-')
plt.xlabel("Share Confirming")
plt.ylabel("Echo Chamber Metric\n(Avg Abs Diff in Beliefs Among Friends)")
plt.title("Impact of Attitude on Echo Chamber Emergence")
plt.show()

# Rationale:
# This script tests how predisposition to confirm (or not) affects the similarity of beliefs among friends.
# We expect that a higher share of confirming agents (e.g. 0.8) yields lower average differences (i.e. stronger echo chambers)
# than a lower share (e.g. 0.2).
