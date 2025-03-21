#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
# Import your updated DisasterModel and HumanAgent from your module.
from DisasterModelNew import DisasterModel, HumanAgent

############################################
# Monkey-patch HumanAgent to record AI reports.
############################################
if not hasattr(HumanAgent, "_ai_patched"):
    original_init = HumanAgent.__init__
    def patched_init(self, unique_id, model, id_num, agent_type="exploitative", share_confirming=0.6):
        original_init(self, unique_id, model, id_num, agent_type, share_confirming)
        self.ai_reported = {}  # dictionary: cell -> list of AI provided values
    HumanAgent.__init__ = patched_init
    HumanAgent._ai_patched = True

if not hasattr(HumanAgent, "_ai_request_patched"):
    original_request_information = HumanAgent.request_information
    def patched_request_information(self):
        original_request_information(self)
        # If an AI candidate was used and no report was recorded, use a fallback.
        # (This fallback simply records the current belief for each cell if no AI report was captured.)
        if not self.ai_reported and hasattr(self, "calls_ai") and self.calls_ai > 0:
            for cell in self.beliefs:
                self.ai_reported.setdefault(cell, []).append(self.beliefs[cell])
    HumanAgent.request_information = patched_request_information
    HumanAgent._ai_request_patched = True

############################################
# Define additional metric: AI Echo Chamber Metric.
############################################
def compute_ai_echo_chamber_metric(model):
    differences = []
    # For each human agent...
    for agent in model.humans.values():
        # Determine the subset of cells where the agent has sent relief.
        # Here we use pending_relief entries that include a target cell.
        target_cells = set()
        for entry in agent.pending_relief:
            if len(entry) == 5:
                # entry = (tick, source_id, accepted_count, confirmations, target_cell)
                target_cells.add(entry[4])
        # Alternatively, you might use cells where tokens were delivered.
        if target_cells and hasattr(agent, "ai_reported") and agent.ai_reported:
            cell_diffs = []
            for cell in target_cells:
                if cell in agent.ai_reported:
                    avg_ai = np.mean(agent.ai_reported[cell])
                    diff = abs(agent.beliefs[cell] - avg_ai)
                    cell_diffs.append(diff)
            if cell_diffs:
                differences.append(np.mean(cell_diffs))
    return np.mean(differences) if differences else None

############################################
# Standard metrics from simulation.
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

def run_simulation(share_exploitative, num_ticks=200):
    model = DisasterModel(
        share_exploitative=share_exploitative,  # Vary this parameter (e.g., 0.2, 0.5, 0.8)
        share_of_disaster=0.2,
        initial_trust=0.5,
        initial_ai_trust=0.75,
        number_of_humans=50,
        share_confirming=0.5,  # fixed
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
    human_echo = compute_echo_chamber_metric(model)
    ai_echo = compute_ai_echo_chamber_metric(model)
    assisted_in_need, assisted_incorrect = compute_assistance_metrics(model)
    return human_echo, ai_echo, assisted_in_need, assisted_incorrect

############################################
# Experiment: Vary share_exploitative
############################################
share_exploitative_values = [0.2, 0.5, 0.8]
num_runs = 10

# Dictionaries to store lists of metrics for each parameter value.
human_echo_data = {}
ai_echo_data = {}
need_data = {}
incorrect_data = {}

for se in share_exploitative_values:
    h_echo_list = []
    ai_echo_list = []
    need_list = []
    incorrect_list = []
    for run in range(num_runs):
        h_echo, ai_echo, need, incorrect = run_simulation(se, num_ticks=200)
        h_echo_list.append(h_echo)
        ai_echo_list.append(ai_echo)
        need_list.append(need)
        incorrect_list.append(incorrect)
    human_echo_data[se] = h_echo_list
    ai_echo_data[se] = ai_echo_list
    need_data[se] = need_list
    incorrect_data[se] = incorrect_list

############################################
# Helper function to compute mean, 25th, and 75th percentiles.
############################################
def compute_stats(data_list):
    data = np.array(data_list)
    mean = np.mean(data)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    return mean, abs(mean - p25), abs(p75 - mean)

def prepare_plot_data(param_values, data_dict):
    means = []
    lower = []  # error from mean to 25th (absolute value)
    upper = []  # error from mean to 75th (absolute value)
    for p in param_values:
        mean, err_low, err_high = compute_stats(data_dict[p])
        means.append(mean)
        lower.append(err_low)
        upper.append(err_high)
    return means, lower, upper

x = share_exploitative_values

human_echo_means, human_echo_lower, human_echo_upper = prepare_plot_data(x, human_echo_data)
ai_echo_means, ai_echo_lower, ai_echo_upper = prepare_plot_data(x, ai_echo_data)
need_means, need_lower, need_upper = prepare_plot_data(x, need_data)
incorrect_means, incorrect_lower, incorrect_upper = prepare_plot_data(x, incorrect_data)

plt.figure(figsize=(16,4))
plt.subplot(1,4,1)
plt.errorbar(x, human_echo_means, yerr=[human_echo_lower, human_echo_upper], fmt='o-', capsize=5)
plt.xlabel("Share Exploitative")
plt.ylabel("Human Echo Chamber Metric")
plt.title("Human Echo vs. Share Exploitative")

plt.subplot(1,4,2)
plt.errorbar(x, ai_echo_means, yerr=[ai_echo_lower, ai_echo_upper], fmt='o-', capsize=5, color='orange')
plt.xlabel("Share Exploitative")
plt.ylabel("AI Echo Chamber Metric")
plt.title("AI Echo vs. Share Exploitative")

plt.subplot(1,4,3)
plt.errorbar(x, need_means, yerr=[need_lower, need_upper], fmt='o-', capsize=5, color='green')
plt.xlabel("Share Exploitative")
plt.ylabel("Cells in Need Assisted")
plt.title("Assistance in Need vs. Share Exploitative")

plt.subplot(1,4,4)
plt.errorbar(x, incorrect_means, yerr=[incorrect_lower, incorrect_upper], fmt='o-', capsize=5, color='red')
plt.xlabel("Share Exploitative")
plt.ylabel("Incorrect Assistance")
plt.title("Incorrect Assistance vs. Share Exploitative")

plt.tight_layout()
plt.show()
