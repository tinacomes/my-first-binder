import numpy as np
import matplotlib.pyplot as plt
import random
import math
import networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from DisasterModelNew import DisasterModel

# --- Modified AIAgent to include an ai_adaptation flag ---
class AIAgent(Agent):
    def __init__(self, unique_id, model, ai_adaptation=True):
        super().__init__(model)
        self.unique_id = unique_id
        self.model = model
        self.ai_adaptation = ai_adaptation  # if False, no adaptation is applied.
        self.memory = {}
        self.sensed = {}

    def sense_environment(self):
        num_cells = int(0.1 * self.model.width * self.model.height)
        self.sensed = {}
        cells = random.sample(list(self.model.disaster_grid.keys()), num_cells)
        for cell in cells:
            self.sensed[cell] = self.model.disaster_grid[cell]

    def provide_information_full(self, human_beliefs, trust):
        info = {}
        for cell, sensed_val in self.sensed.items():
            human_val = human_beliefs.get(cell, sensed_val)
            if self.ai_adaptation:
                if abs(sensed_val - human_val) > 1:
                    correction_factor = 1 - min(1, trust)
                    corrected = round(sensed_val + correction_factor * (human_val - sensed_val))
                else:
                    corrected = sensed_val
            else:
                corrected = sensed_val
            info[cell] = corrected
        return info

    def step(self):
        self.sense_environment()

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

def run_simulation(ai_adaptation, num_ticks=300):
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
    # Replace AI agents with our modified version.
    model.ais = {}
    for k in range(model.num_ai):
        a = AIAgent(unique_id=f"A_{k}", model=model, ai_adaptation=ai_adaptation)
        model.ais[f"A_{k}"] = a
        model.schedule.add(a)
        x = random.randrange(model.width)
        y = random.randrange(model.height)
        model.grid.place_agent(a, (x, y))
    for t in range(num_ticks):
        model.step()
    echo_metric = compute_echo_chamber_metric(model)
    assisted_in_need, assisted_incorrect = compute_assistance_metrics(model)
    return echo_metric, assisted_in_need, assisted_incorrect

num_runs = 5
adapt_echo = []
adapt_need = []
adapt_incorrect = []
noadapt_echo = []
noadapt_need = []
noadapt_incorrect = []

for _ in range(num_runs):
    echo1, need1, incorrect1 = run_simulation(ai_adaptation=True)
    adapt_echo.append(echo1)
    adapt_need.append(need1)
    adapt_incorrect.append(incorrect1)
    echo2, need2, incorrect2 = run_simulation(ai_adaptation=False)
    noadapt_echo.append(echo2)
    noadapt_need.append(need2)
    noadapt_incorrect.append(incorrect2)

mean_adapt_echo = np.mean(adapt_echo)
mean_noadapt_echo = np.mean(noadapt_echo)
mean_adapt_need = np.mean(adapt_need)
mean_noadapt_need = np.mean(noadapt_need)
mean_adapt_incorrect = np.mean(adapt_incorrect)
mean_noadapt_incorrect = np.mean(noadapt_incorrect)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(["AI Adapts", "AI No Adapt"], [mean_adapt_echo, mean_noadapt_echo], color=["lightblue", "salmon"])
plt.ylabel("Echo Chamber Metric")
plt.title("Impact of AI Adaptation on Echo Chambers")

plt.subplot(1,2,2)
x_labels = ["Need Assisted", "Incorrect Assistance"]
adapt_vals = [mean_adapt_need, mean_adapt_incorrect]
noadapt_vals = [mean_noadapt_need, mean_noadapt_incorrect]
x = np.arange(len(x_labels))
width = 0.35
plt.bar(x - width/2, adapt_vals, width, label="AI Adapts", color="lightblue")
plt.bar(x + width/2, noadapt_vals, width, label="AI No Adapt", color="salmon")
plt.xticks(x, x_labels)
plt.ylabel("Number of Grid Cells")
plt.title("Assistance Metrics by AI Behavior")
plt.legend()
plt.tight_layout()
plt.show()

# Rationale:
# This experiment compares two modes of AI behavior. If AI agents adapt their output to human beliefs,
# they may help to break echo chambers (i.e. produce a higher echo chamber metric indicating more diversity among friends)
# and change the pattern of assistance (potentially assisting more cells in need and reducing incorrect assistance).
