#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import random, math, networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from DisasterModelNew import DisasterModel, HumanAgent

############################################
# Monkey-patch HumanAgent to record AI reports (only once)
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
        # In a proper AI branch, AI-provided info should be recorded.
        # As a fallback, if calls to AI occurred and no report was recorded, record current beliefs.
        if not self.ai_reported and hasattr(self, "calls_ai") and self.calls_ai > 0:
            for cell in self.beliefs:
                self.ai_reported.setdefault(cell, []).append(self.beliefs[cell])
    HumanAgent.request_information = patched_request_information
    HumanAgent._ai_request_patched = True

############################################
# Update AI Echo Chamber Metric: weight by average AI trust.
############################################
def compute_ai_echo_chamber_metric(model):
    differences = []
    for agent in model.humans.values():
        # Gather the set of target cells (cells for which the agent has pending relief)
        target_cells = set()
        for entry in agent.pending_relief:
            if len(entry) == 5:
                target_cells.add(entry[4])
        # If no target cells, skip.
        if not target_cells:
            continue
        # For each target cell where AI reports exist, compute the difference.
        cell_diffs = []
        for cell in target_cells:
            if cell in agent.ai_reported:
                avg_ai = np.mean(agent.ai_reported[cell])
                diff = abs(agent.beliefs[cell] - avg_ai)
                cell_diffs.append(diff)
        if cell_diffs:
            # Weight the agent’s deviation by its average AI trust.
            ai_trusts = [v for key, v in agent.trust.items() if key.startswith("A_")]
            avg_ai_trust = np.mean(ai_trusts) if ai_trusts else 1
            differences.append(np.mean(cell_diffs) * avg_ai_trust)
    return np.mean(differences) if differences else None

############################################
# Standard metrics.
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

############################################
# Run one simulation and record time-series metrics.
############################################
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
    ticks = []
    human_echo_ts = []
    ai_echo_ts = []
    trust_ai_ts = []   # average AI trust across agents
    unmet_needs_ts = []
    # We also record assistance metrics over time (cells in need assisted)
    assisted_need_ts = []
    assisted_incorrect_ts = []
    
    for t in range(num_ticks):
        model.step()
        ticks.append(t)
        # Compute echo chamber metrics at this tick.
        h_echo = compute_echo_chamber_metric(model)
        a_echo = compute_ai_echo_chamber_metric(model)
        human_echo_ts.append(h_echo)
        ai_echo_ts.append(a_echo)
        # Compute average AI trust (over all agents) for information sources.
        ai_trust_list = []
        for agent in model.humans.values():
            ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
            if ai_vals:
                ai_trust_list.append(np.mean(ai_vals))
        trust_ai_ts.append(np.mean(ai_trust_list) if ai_trust_list else 0)
        # Unmet needs.
        unmet = 0
        for pos, level in model.disaster_grid.items():
            if level >= 4 and model.tokens_this_tick.get(pos, 0) == 0:
                unmet += 1
        unmet_needs_ts.append(unmet)
        # Assistance metrics.
        need, incorrect = compute_assistance_metrics(model)
        assisted_need_ts.append(need)
        assisted_incorrect_ts.append(incorrect)
    return ticks, human_echo_ts, ai_echo_ts, trust_ai_ts, unmet_needs_ts, assisted_need_ts, assisted_incorrect_ts

############################################
# Aggregate multiple runs.
############################################
num_runs = 5
num_ticks = 300

all_human_echo = []
all_ai_echo = []
all_trust_ai = []
all_unmet = []
all_assist_need = []
all_assist_incorrect = []

for run in range(num_runs):
    ticks, h_echo_ts, a_echo_ts, trust_ai_ts, unmet_ts, assist_need_ts, assist_incorrect_ts = run_model_collect_timeseries(num_ticks)
    all_human_echo.append(h_echo_ts)
    all_ai_echo.append(a_echo_ts)
    all_trust_ai.append(trust_ai_ts)
    all_unmet.append(unmet_ts)
    all_assist_need.append(assist_need_ts)
    all_assist_incorrect.append(assist_incorrect_ts)

all_human_echo = np.array(all_human_echo)
all_ai_echo = np.array(all_ai_echo)
all_trust_ai = np.array(all_trust_ai)
all_unmet = np.array(all_unmet)
all_assist_need = np.array(all_assist_need)
all_assist_incorrect = np.array(all_assist_incorrect)
ticks_arr = np.array(ticks)

def aggregate_timeseries(data):
    mean = np.mean(data, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    return mean, np.abs(mean - p25), np.abs(p75 - mean)

human_echo_mean, human_echo_err_lower, human_echo_err_upper = aggregate_timeseries(all_human_echo)
ai_echo_mean, ai_echo_err_lower, ai_echo_err_upper = aggregate_timeseries(all_ai_echo)
trust_ai_mean, trust_ai_err_lower, trust_ai_err_upper = aggregate_timeseries(all_trust_ai)
unmet_mean, unmet_err_lower, unmet_err_upper = aggregate_timeseries(all_unmet)
assist_need_mean, assist_need_err_lower, assist_need_err_upper = aggregate_timeseries(all_assist_need)
assist_incorrect_mean, assist_incorrect_err_lower, assist_incorrect_err_upper = aggregate_timeseries(all_assist_incorrect)

############################################
# Plot time-series with error bands.
############################################
plt.figure(figsize=(16,12))

# 1. Human Echo Chamber Metric over time.
plt.subplot(3,2,1)
plt.plot(ticks_arr, human_echo_mean, label="Human Echo", color='blue')
plt.fill_between(ticks_arr, human_echo_mean - human_echo_err_lower, human_echo_mean + human_echo_err_upper, color='blue', alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Human Echo Chamber Metric")
plt.title("Human Echo Chamber Over Time")
plt.legend()

# 2. AI Echo Chamber Metric over time.
plt.subplot(3,2,2)
plt.plot(ticks_arr, ai_echo_mean, label="AI Echo", color='orange')
plt.fill_between(ticks_arr, ai_echo_mean - ai_echo_err_lower, ai_echo_mean + ai_echo_err_upper, color='orange', alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("AI Echo Chamber Metric")
plt.title("AI Echo Chamber Over Time")
plt.legend()

# 3. Average Trust in AI over time.
plt.subplot(3,2,3)
plt.plot(ticks_arr, trust_ai_mean, label="Avg AI Trust", color='green')
plt.fill_between(ticks_arr, trust_ai_mean - trust_ai_err_lower, trust_ai_mean + trust_ai_err_upper, color='green', alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Average Trust in AI")
plt.title("Trust in AI Over Time")
plt.legend()

# 4. Unmet Needs over time.
plt.subplot(3,2,4)
plt.plot(ticks_arr, unmet_mean, label="Unmet Needs", color='red', marker='o', markersize=3)
plt.fill_between(ticks_arr, unmet_mean - unmet_err_lower, unmet_mean + unmet_err_upper, color='red', alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Unmet Needs (Cells)")
plt.title("Unmet Needs Over Time")
plt.legend()

# 5. Assistance in Need over time.
plt.subplot(3,2,5)
plt.plot(ticks_arr, assist_need_mean, label="Cells in Need Assisted", color='purple')
plt.fill_between(ticks_arr, assist_need_mean - assist_need_err_lower, assist_need_mean + assist_need_err_upper, color='purple', alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Cells Assisted")
plt.title("Assistance in Need Over Time")
plt.legend()

# 6. Incorrect Assistance over time.
plt.subplot(3,2,6)
plt.plot(ticks_arr, assist_incorrect_mean, label="Incorrect Assistance", color='brown')
plt.fill_between(ticks_arr, assist_incorrect_mean - assist_incorrect_err_lower, assist_incorrect_mean + assist_incorrect_err_upper, color='brown', alpha=0.3)
plt.xlabel("Tick")
plt.ylabel("Cells Assisted Incorrectly")
plt.title("Incorrect Assistance Over Time")
plt.legend()

plt.tight_layout()
plt.show()
