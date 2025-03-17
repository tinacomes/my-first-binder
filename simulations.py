import numpy as np
import matplotlib.pyplot as plt
from DisasterOpinion import DisasterModel

# Define the parameter scenarios.
exploitative_shares = [0.3, 0.5, 0.7]      # Fraction of exploitative agents.
confirming_shares = [0.3, 0.5, 0.7]         # Fraction of confirming agents.
disaster_dynamics_vals = [1, 2, 3]          # Different disaster dynamics.

# For each scenario, run a number of simulation runs.
runs_per_scenario = 5
ticks = 30

# Containers for results.
scenario_results_unmet = []  # For unmet needs.
scenario_results_trust = []  # For trust in AI vs humans.
scenario_results_calls = []  # For calls to AI vs humans.
scenario_labels = []

for exp_share in exploitative_shares:
    for conf_share in confirming_shares:
        for dynamics in disaster_dynamics_vals:
            unmet_all = []
            trust_all = []  # We'll store tuple (avg_human_trust, avg_ai_trust) averaged over all agents.
            calls_all = []  # Tuple (calls_human, calls_ai)
            for run in range(runs_per_scenario):
                model = DisasterModel(share_exploitative=exp_share,
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
                unmet_all.append(model.unmet_needs_evolution[-1])
                # Average trust for human sources and AI sources across all agents.
                human_trust = []
                ai_trust = []
                calls_human = []
                calls_ai = []
                for agent in model.humans.values():
                    human_vals = [v for key, v in agent.trust.items() if key.startswith("H_")]
                    ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
                    if human_vals:
                        human_trust.append(sum(human_vals)/len(human_vals))
                    if ai_vals:
                        ai_trust.append(sum(ai_vals)/len(ai_vals))
                    calls_human.append(agent.calls_human)
                    calls_ai.append(agent.calls_ai)
                trust_all.append((np.mean(human_trust), np.mean(ai_trust)))
                calls_all.append((np.mean(calls_human), np.mean(calls_ai)))
            scenario_results_unmet.append(unmet_all)
            # Compute trust differences.
            trust_diff = [t[1] - t[0] for t in trust_all]
            scenario_results_trust.append(trust_diff)
            # Compute call differences.
            calls_diff = [c[1] - c[0] for c in calls_all]
            scenario_results_calls.append(calls_diff)
            label = f"Exp:{exp_share}, Conf:{conf_share}, Dyn:{dynamics}"
            scenario_labels.append(label)

# Plot box plots for unmet needs.
plt.figure(figsize=(12, 6))
plt.boxplot(scenario_results_unmet, labels=scenario_labels, showmeans=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Unmet Needs (cells with level ≥ 4 with no assistance)")
plt.title("Box Plot of Unmet Needs across Scenarios")
plt.tight_layout()
plt.show()

# Plot box plots for trust difference (AI trust minus Human trust).
plt.figure(figsize=(12, 6))
plt.boxplot(scenario_results_trust, labels=scenario_labels, showmeans=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("AI Trust - Human Trust (average across agents)")
plt.title("Box Plot of Trust Differences across Scenarios")
plt.tight_layout()
plt.show()

# Plot box plots for calls difference (calls to AI minus calls to Humans).
plt.figure(figsize=(12, 6))
plt.boxplot(scenario_results_calls, labels=scenario_labels, showmeans=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Calls to AI - Calls to Humans (average across agents)")
plt.title("Box Plot of Information Request Differences across Scenarios")
plt.tight_layout()
plt.show()
