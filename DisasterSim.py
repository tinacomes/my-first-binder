#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from DisasterModelNew import DisasterModel

def run_model_collect_timeseries(num_ticks=300):
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
    # Prepare lists to record time series.
    ticks = []
    trust_data = []      # will be a list of tuples per tick: (exp_human, exp_ai, expl_human, expl_ai)
    unmet_needs = []
    calls_data = []      # list of tuples per tick: (calls_exp_human, calls_exp_ai, calls_expl_human, calls_expl_ai)
    
    for t in range(num_ticks):
        model.step()
        ticks.append(t)
        # Record trust: compute averages per agent type.
        exp_human = []
        exp_ai = []
        expl_human = []
        expl_ai = []
        for agent in model.humans.values():
            human_trust = [v for key, v in agent.trust.items() if key.startswith("H_")]
            ai_trust = [v for key, v in agent.trust.items() if key.startswith("A_")]
            if agent.agent_type == "exploitative":
                if human_trust:
                    exp_human.append(np.mean(human_trust))
                if ai_trust:
                    exp_ai.append(np.mean(ai_trust))
            else:
                if human_trust:
                    expl_human.append(np.mean(human_trust))
                if ai_trust:
                    expl_ai.append(np.mean(ai_trust))
        trust_data.append((
            np.mean(exp_human) if exp_human else 0,
            np.mean(exp_ai) if exp_ai else 0,
            np.mean(expl_human) if expl_human else 0,
            np.mean(expl_ai) if expl_ai else 0
        ))
        # Unmet needs: count grid cells with level>=4 that got no token in this tick.
        unmet = 0
        for pos, level in model.disaster_grid.items():
            if level >= 4 and model.tokens_this_tick.get(pos, 0) == 0:
                unmet += 1
        unmet_needs.append(unmet)
        # Record calls data.
        calls_data.append(tuple(model.calls_data[-1]))
    return ticks, trust_data, unmet_needs, calls_data

# Run multiple simulations and aggregate per tick.
num_runs = 5
num_ticks = 300

# Arrays to collect time-series data from each run.
all_trust = []   # shape (num_runs, num_ticks, 4)
all_unmet = []   # shape (num_runs, num_ticks)
all_calls = []   # shape (num_runs, num_ticks, 4)

for run in range(num_runs):
    ticks, trust_ts, unmet_ts, calls_ts = run_model_collect_timeseries(num_ticks)
    all_trust.append(trust_ts)
    all_unmet.append(unmet_ts)
    all_calls.append(calls_ts)
all_trust = np.array(all_trust)
all_unmet = np.array(all_unmet)
all_calls = np.array(all_calls)

# For each tick, compute mean and 25th/75th percentiles.
def aggregate_timeseries(data):
    mean = np.mean(data, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    return mean, p25, p75

trust_mean, trust_p25, trust_p75 = aggregate_timeseries(all_trust)  # shape (num_ticks,4)
unmet_mean, unmet_p25, unmet_p75 = aggregate_timeseries(all_unmet)  # shape (num_ticks,)
calls_mean, calls_p25, calls_p75 = aggregate_timeseries(all_calls)  # shape (num_ticks,4)

ticks_arr = np.array(ticks)

plt.figure(figsize=(16,10))
# Trust evolution plot.
plt.subplot(2,2,1)
labels = ["Exploitative-Human", "Exploitative-AI", "Exploratory-Human", "Exploratory-AI"]
for i in range(4):
    plt.plot(ticks_arr, trust_mean[:, i], label=labels[i])
    plt.fill_between(ticks_arr, trust_p25[:, i], trust_p75[:, i], alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Average Trust")
plt.title("Trust Evolution by Agent Type")
plt.legend()

# Unmet needs time series.
plt.subplot(2,2,2)
plt.plot(ticks_arr, unmet_mean, marker='o', markersize=3)
plt.fill_between(ticks_arr, unmet_p25, unmet_p75, alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Unmet Needs (Cells with level>=4, no tokens)")
plt.title("Unmet Needs Evolution")

# Calls data time series.
plt.subplot(2,2,3)
call_labels = ["Calls Exp-Human", "Calls Exp-AI", "Calls Expl-Human", "Calls Expl-AI"]
for i in range(4):
    plt.plot(ticks_arr, calls_mean[:, i], label=call_labels[i])
    plt.fill_between(ticks_arr, calls_p25[:, i], calls_p75[:, i], alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Information Requests")
plt.title("Information Request Calls by Agent Type")
plt.legend()

plt.tight_layout()
plt.show()
