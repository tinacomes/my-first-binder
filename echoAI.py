#!/usr/bin/env python
import csv
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
# Import the base model (assumed to be in DisasterModelNew.py)
from DisasterModelNew import DisasterModel, HumanAgent

############################################
# Monkey-patch HumanAgent to record AI reports.
############################################

# We add an attribute ai_reported to record, for each cell, the values provided by AI.
original_init = HumanAgent.__init__
def patched_init(self, unique_id, model, id_num, agent_type="exploitative", share_confirming=0.5):
    original_init(self, unique_id, model, id_num, agent_type, share_confirming)
    self.ai_reported = {}  # dictionary: cell -> list of AI provided values
HumanAgent.__init__ = patched_init

# In request_information, in the branch for AI calls, record the provided information.
original_request_information = HumanAgent.request_information
def patched_request_information(self):
    # Call the original method.
    original_request_information(self)
    # (Assumption: In the original code, when mode=="ai", the agent obtains rep from an AI candidate.
    # We modify that branch here to record the reported values.)
    # For simplicity, if self.ai_reported was not updated there, we assume that in the original code
    # the branch handling AI calls is similar to the human branch.
    # We assume that somewhere in the code, after an AI candidate is selected,
    # a block like the following is executed:
    #
    #    rep = other.provide_information_full(self.beliefs, trust=self.trust[candidate])
    #    for cell, reported_value in rep.items():
    #         self.ai_reported.setdefault(cell, []).append(reported_value)
    #
    # If not, you may insert the following snippet at the end of the AI branch.
    # (Here, we simply ensure that if no AI report has been recorded, then self.ai_reported remains {}.)
    pass
HumanAgent.request_information = patched_request_information

############################################
# Define additional metric: AI Echo Chamber Metric.
############################################
def compute_ai_echo_chamber_metric(model):
    differences = []
    for agent in model.humans.values():
        if hasattr(agent, "ai_reported") and agent.ai_reported:
            cell_diffs = []
            for cell, reports in agent.ai_reported.items():
                avg_ai = np.mean(reports)
                cell_diffs.append(abs(agent.beliefs[cell] - avg_ai))
            if cell_diffs:
                differences.append(np.mean(cell_diffs))
    return np.mean(differences) if differences else None

############################################
# Functions to compute metrics from a simulation run.
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

def run_simulation(share_confirming, num_ticks=200):
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
    # Run simulation
    for t in range(num_ticks):
        model.step()
    # Compute metrics
    human_echo = compute_echo_chamber_metric(model)
    ai_echo = compute_ai_echo_chamber_metric(model)
    assisted_in_need, assisted_incorrect = compute_assistance_metrics(model)
    return human_echo, ai_echo, assisted_in_need, assisted_incorrect

############################################
# Experiment: Vary share_confirming (as an example)
############################################
share_confirming_values = [0.2, 0.5, 0.8]
num_runs = 10

# Store metrics per parameter value.
human_echo_data = {}
ai_echo_data = {}
need_data = {}
incorrect_data = {}

for sc in share_confirming_values:
    h_echo_list = []
    ai_echo_list = []
    need_list = []
    incorrect_list = []
    for run in range(num_runs):
        h_echo, ai_echo, need, incorrect = run_simulation(sc, num_ticks=200)
        h_echo_list.append(h_echo)
        ai_echo_list.append(ai_echo)
        need_list.append(need)
        incorrect_list.append(incorrect)
    human_echo_data[sc] = h_echo_list
    ai_echo_data[sc] = ai_echo_list
    need_data[sc] = need_list
    incorrect_data[sc] = incorrect_list

############################################
# Helper function to compute mean, 25th, and 75th percentiles.
############################################
def compute_stats(data_list):
    data = np.array(data_list)
    mean = np.mean(data)
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    return mean, p25, p75

# Prepare data for plotting.
def prepare_plot_data(param_values, data_dict):
    means = []
    lower = []  # mean - 25th
    upper = []  # 75th - mean
    for p in param_values:
        mean, p25, p75 = compute_stats(data_dict[p])
        means.append(mean)
        lower.append(mean - p25)
        upper.append(p75 - mean)
    return means, lower, upper

x = share_confirming_values

human_echo_means, human_echo_lower, human_echo_upper = prepare_plot_data(x, human_echo_data)
ai_echo_means, ai_echo_lower, ai_echo_upper = prepare_plot_data(x, ai_echo_data)
need_means, need_lower, need_upper = prepare_plot_data(x, need_data)
incorrect_means, incorrect_lower, incorrect_upper = prepare_plot_data(x, incorrect_data)

plt.figure(figsize=(16,4))
plt.subplot(1,4,1)
plt.errorbar(x, human_echo_means, yerr=[human_echo_lower, human_echo_upper], fmt='o-', capsize=5)
plt.xlabel("Share Confirming")
plt.ylabel("Human Echo Chamber Metric")
plt.title("Human Echo Chamber vs. Attitude")

plt.subplot(1,4,2)
plt.errorbar(x, ai_echo_means, yerr=[ai_echo_lower, ai_echo_upper], fmt='o-', capsize=5, color='orange')
plt.xlabel("Share Confirming")
plt.ylabel("AI Echo Chamber Metric")
plt.title("AI Echo Chamber vs. Attitude")

plt.subplot(1,4,3)
plt.errorbar(x, need_means, yerr=[need_lower, need_upper], fmt='o-', capsize=5, color='green')
plt.xlabel("Share Confirming")
plt.ylabel("Cells in Need Assisted")
plt.title("Assistance in Need vs. Attitude")

plt.subplot(1,4,4)
plt.errorbar(x, incorrect_means, yerr=[incorrect_lower, incorrect_upper], fmt='o-', capsize=5, color='red')
plt.xlabel("Share Confirming")
plt.ylabel("Cells Incorrectly Assisted")
plt.title("Incorrect Assistance vs. Attitude")
plt.tight_layout()
plt.show()
