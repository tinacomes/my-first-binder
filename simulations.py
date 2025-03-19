import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from DisasterOpinion import DisasterModel

# -----------------------------
# Settings for simulation experiments
# -----------------------------
exploitative_shares = [0.3, 0.5, 0.7]      # Fraction of exploitative agents.
confirming_shares = [0.3, 0.5, 0.7]         # Fraction of confirming agents.
disaster_dynamics_vals = [1, 2, 3]          # Different disaster dynamics.

runs_per_scenario = 5
ticks = 500  # Increased number of ticks

# Create a folder to store run results.
results_dir = "simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# -----------------------------
# Run simulations and save results for each run
# -----------------------------
scenario_keys = []  # to store scenario keys
for exp_share in exploitative_shares:
    for conf_share in confirming_shares:
        for dynamics in disaster_dynamics_vals:
            scenario_key = f"exp_{exp_share}_conf_{conf_share}_dyn_{dynamics}"
            scenario_keys.append(scenario_key)
            for run in range(runs_per_scenario):
                # Create and run the model
                model = DisasterModel(
                    share_exploitative=exp_share,
                    share_of_disaster=0.2,
                    initial_trust=0.7,
                    initial_ai_trust=0.7,
                    number_of_humans=50,
                    share_confirming=conf_share,
                    disaster_dynamics=dynamics,
                    shock_probability=0.1,
                    shock_magnitude=2,
                    trust_update_mode="average",
                    exploitative_correction_factor=1.0,
                    width=50,
                    height=50)
                for t in range(ticks):
                    model.step()
                # Record final unmet needs (from the last tick).
                unmet_final = model.unmet_needs_evolution[-1]
                # Compute average trust and call data separately for exploitative and exploratory agents.
                exp_human_trust = []
                exp_ai_trust = []
                expl_human_trust = []
                expl_ai_trust = []
                calls_human = []
                calls_ai = []
                for agent in model.humans.values():
                    if agent.agent_type == "exploitative":
                        human_vals = [v for key, v in agent.trust.items() if key.startswith("H_")]
                        ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
                        if human_vals:
                            exp_human_trust.append(np.mean(human_vals))
                        if ai_vals:
                            exp_ai_trust.append(np.mean(ai_vals))
                        calls_human.append(agent.calls_human)
                        calls_ai.append(agent.calls_ai)
                    else:
                        human_vals = [v for key, v in agent.trust.items() if key.startswith("H_")]
                        ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
                        if human_vals:
                            expl_human_trust.append(np.mean(human_vals))
                        if ai_vals:
                            expl_ai_trust.append(np.mean(ai_vals))
                        calls_human.append(agent.calls_human)
                        calls_ai.append(agent.calls_ai)
                exp_trust = (np.mean(exp_human_trust) if exp_human_trust else np.nan,
                             np.mean(exp_ai_trust) if exp_ai_trust else np.nan)
                expl_trust = (np.mean(expl_human_trust) if expl_human_trust else np.nan,
                              np.mean(expl_ai_trust) if expl_ai_trust else np.nan)
                calls_avg = (np.mean(calls_human), np.mean(calls_ai))
                # Get the trust evolution over time (model.trust_data is a list of 4-tuples).
                trust_evolution = np.array(model.trust_data)  # shape: (ticks, 4)
                # Save results to compressed file.
                filename = os.path.join(results_dir, f"{scenario_key}_run_{run}.npz")
                np.savez_compressed(filename, unmet=unmet_final,
                                    exp_trust=exp_trust, expl_trust=expl_trust,
                                    calls_avg=calls_avg, trust_evolution=trust_evolution)
                print(f"Saved {filename}")
                # Clean up to free memory.
                del model
                gc.collect()

# -----------------------------
# Aggregate results across runs for each scenario without loading everything at once.
# -----------------------------
# Prepare dictionaries to hold aggregated data.
all_scenarios_unmet = {}
all_scenarios_exp_trust_evo = {}   # will hold list of arrays (shape: ticks x 2) per run for exploitative agents.
all_scenarios_expl_trust_evo = {}  # similarly for exploratory agents.
all_scenarios_calls = {}

for key in scenario_keys:
    unmet_list = []
    exp_trust_evo_list = []  # each element: (ticks, 2) array for exploitative agents (human, AI)
    expl_trust_evo_list = []  # each element: (ticks, 2) array for exploratory agents
    calls_list = []
    for run in range(runs_per_scenario):
        filename = os.path.join(results_dir, f"{key}_run_{run}.npz")
        with np.load(filename) as data:
            unmet_list.append(data["unmet"])
            trust_evo = data["trust_evolution"]  # shape (ticks, 4)
            # Split trust evolution: first 2 columns for exploitative, last 2 for exploratory.
            exp_trust_evo_list.append(trust_evo[:, :2])
            expl_trust_evo_list.append(trust_evo[:, 2:])
            calls_list.append(data["calls_avg"])
    all_scenarios_unmet[key] = np.array(unmet_list)
    all_scenarios_exp_trust_evo[key] = np.array(exp_trust_evo_list)   # shape (runs, ticks, 2)
    all_scenarios_expl_trust_evo[key] = np.array(expl_trust_evo_list)   # shape (runs, ticks, 2)
    all_scenarios_calls[key] = np.array(calls_list)

# -----------------------------
# Produce Box Plots for Final Metrics
# -----------------------------
scenario_labels = list(all_scenarios_unmet.keys())

# Box plot for final unmet needs.
unmet_data = [all_scenarios_unmet[key] for key in scenario_labels]
plt.figure(figsize=(12, 6))
plt.boxplot(unmet_data, labels=scenario_labels, showmeans=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Final Unmet Needs")
plt.title("Box Plot of Final Unmet Needs across Scenarios")
plt.tight_layout()
plt.show()

# Box plot for trust difference (exploitative agents: AI trust minus Human trust at final tick).
trust_diff_data = []
for key in scenario_labels:
    exp_trust_final = all_scenarios_exp_trust_evo[key][:, -1, :]  # shape (runs, 2)
    diff = exp_trust_final[:, 1] - exp_trust_final[:, 0]
    trust_diff_data.append(diff)
plt.figure(figsize=(12, 6))
plt.boxplot(trust_diff_data, labels=scenario_labels, showmeans=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Final AI Trust - Human Trust (Exploitative)")
plt.title("Box Plot of Trust Differences across Scenarios")
plt.tight_layout()
plt.show()

# Box plot for calls difference (calls to AI minus calls to Humans).
calls_diff_data = []
for key in scenario_labels:
    calls_avg = all_scenarios_calls[key]  # shape (runs, 2)
    diff = calls_avg[:, 1] - calls_avg[:, 0]
    calls_diff_data.append(diff)
plt.figure(figsize=(12, 6))
plt.boxplot(calls_diff_data, labels=scenario_labels, showmeans=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Calls to AI - Calls to Humans")
plt.title("Box Plot of Information Request Differences across Scenarios")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot Evolution of Trust over Time with 90%-Percentile Corridors
# -----------------------------
ticks_array = np.arange(ticks)

# Exploitative agents trust evolution.
plt.figure(figsize=(12, 6))
for key in scenario_labels:
    data = all_scenarios_exp_trust_evo[key]  # shape (runs, ticks, 2)
    # Compute mean and 5th/95th percentiles for human and AI trust separately.
    mean_human = np.mean(data[:, :, 0], axis=0)
    mean_ai = np.mean(data[:, :, 1], axis=0)
    perc5_human = np.percentile(data[:, :, 0], 5, axis=0)
    perc95_human = np.percentile(data[:, :, 0], 95, axis=0)
    perc5_ai = np.percentile(data[:, :, 1], 5, axis=0)
    perc95_ai = np.percentile(data[:, :, 1], 95, axis=0)
    plt.plot(ticks_array, mean_human, label=f"{key} Exp-Human", linestyle="--")
    plt.fill_between(ticks_array, perc5_human, perc95_human, alpha=0.2)
    plt.plot(ticks_array, mean_ai, label=f"{key} Exp-AI")
    plt.fill_between(ticks_array, perc5_ai, perc95_ai, alpha=0.2)
plt.xlabel("Ticks")
plt.ylabel("Trust Value")
plt.title("Evolution of Trust over Time (Exploitative Agents)")
plt.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

# Exploratory agents trust evolution.
plt.figure(figsize=(12, 6))
for key in scenario_labels:
    data = all_scenarios_expl_trust_evo[key]  # shape (runs, ticks, 2)
    mean_human = np.mean(data[:, :, 0], axis=0)
    mean_ai = np.mean(data[:, :, 1], axis=0)
    perc5_human = np.percentile(data[:, :, 0], 5, axis=0)
    perc95_human = np.percentile(data[:, :, 0], 95, axis=0)
    perc5_ai = np.percentile(data[:, :, 1], 5, axis=0)
    perc95_ai = np.percentile(data[:, :, 1], 95, axis=0)
    plt.plot(ticks_array, mean_human, label=f"{key} Expl-Human", linestyle="--")
    plt.fill_between(ticks_array, perc5_human, perc95_human, alpha=0.2)
    plt.plot(ticks_array, mean_ai, label=f"{key} Expl-AI")
    plt.fill_between(ticks_array, perc5_ai, perc95_ai, alpha=0.2)
plt.xlabel("Ticks")
plt.ylabel("Trust Value")
plt.title("Evolution of Trust over Time (Exploratory Agents)")
plt.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()
