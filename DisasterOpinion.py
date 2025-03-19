import random
import math
import matplotlib.pyplot as plt
import networkx as nx

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

#########################################
# Disaster Model Definition (Updated Trust and Accuracy Version)
#########################################
class DisasterModel(Model):
    def __init__(self,
                 share_exploitative,          # Fraction of humans that are exploitative.
                 share_of_disaster,           # Fraction of grid cells affected initially.
                 initial_trust,               # Baseline trust for human agents.
                 initial_ai_trust,            # Baseline trust for AI agents.
                 number_of_humans,
                 share_confirming,            # Fraction of humans that are "confirming"
                 disaster_dynamics=2,         # Maximum change in disaster per tick.
                 shock_probability=0.1,       # Probability that a shock occurs.
                 shock_magnitude=2,           # Maximum shock magnitude.
                 trust_update_mode="average", # (Not used further here)
                 exploitative_correction_factor=1.0,  # (Not used further)
                 width=50, height=50):
        super().__init__()
        self.share_exploitative = share_exploitative
        self.share_of_disaster = share_of_disaster
        self.base_trust = initial_trust
        self.base_ai_trust = initial_ai_trust
        self.num_humans = number_of_humans
        self.num_ai = 5
        self.share_confirming = share_confirming
        self.width = width
        self.height = height
        self.disaster_dynamics = disaster_dynamics
        self.shock_probability = shock_probability
        self.shock_magnitude = shock_magnitude
        self.trust_update_mode = trust_update_mode
        self.exploitative_correction_factor = exploitative_correction_factor

        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.tick = 0

        # For tracking assistance:
        self.assistance_exploit = {}  # cell -> tokens delivered by exploitative agents
        self.assistance_explor = {}   # cell -> tokens delivered by exploratory agents
        self.unmet_needs_evolution = []  # Number of cells in need (level >= 4) that got no token this tick.

        # Create disaster grid.
        self.disaster_grid = {}
        self.epicenter = (random.randint(0, width - 1), random.randint(0, height - 1))
        total_cells = width * height
        self.disaster_radius = math.sqrt(self.share_of_disaster * total_cells / math.pi)
        for x in range(width):
            for y in range(height):
                d = math.sqrt((x - self.epicenter[0])**2 + (y - self.epicenter[1])**2)
                if d < self.disaster_radius / 3:
                    level = 5
                elif d < 2 * self.disaster_radius / 3:
                    level = 4
                elif d < self.disaster_radius:
                    level = 3
                else:
                    level = 0
                self.disaster_grid[(x, y)] = level

        # Create a Watts–Strogatz network (for friend selection if desired).
        self.social_network = nx.watts_strogatz_graph(self.num_humans, 4, 0.1)

        # Create human agents.
        self.humans = {}
        for i in range(self.num_humans):
            agent_type = "exploitative" if random.random() < self.share_exploitative else "exploratory"
            a = HumanAgent(unique_id=f"H_{i}", model=self, id_num=i, agent_type=agent_type, share_confirming=self.share_confirming)
            self.humans[f"H_{i}"] = a
            self.schedule.add(a)
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(a, (x, y))

        # Initialize trust and info_accuracy.
        for i in range(self.num_humans):
            agent_id = f"H_{i}"
            agent = self.humans[agent_id]
            for j in range(self.num_humans):
                if agent_id == f"H_{j}":
                    continue
                agent.trust[f"H_{j}"] = random.uniform(self.base_trust - 0.05, self.base_trust + 0.05)
                agent.info_accuracy[f"H_{j}"] = random.uniform(0.4, 0.6)
            # Set friend list: randomly select 5% of humans (at least one) and boost trust.
            friend_count = max(1, int(0.05 * self.num_humans))
            possible_friends = [f"H_{j}" for j in range(self.num_humans) if f"H_{j}" != agent_id]
            agent.friends = set(random.sample(possible_friends, friend_count))
            for friend_id in agent.friends:
                agent.trust[friend_id] = min(1, agent.trust[friend_id] + 0.1)
            for k in range(self.num_ai):
                agent.trust[f"A_{k}"] = random.uniform(self.base_ai_trust - 0.1, self.base_ai_trust + 0.1)
                agent.info_accuracy[f"A_{k}"] = random.uniform(0.4, 0.7)

        # Create AI agents.
        self.ais = {}
        for k in range(self.num_ai):
            a = AIAgent(unique_id=f"A_{k}", model=self)
            self.ais[f"A_{k}"] = a
            self.schedule.add(a)
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(a, (x, y))

        self.trust_data = []   # (exploitative-human, exploitative-AI, exploratory-human, exploratory-AI)
        self.calls_data = []   # (# calls to humans, # calls to AI) per group.
        self.rewards_data = [] # (total_reward_exploitative, total_reward_exploratory)

    def update_disaster(self):
        new_grid = {}
        for pos, level in self.disaster_grid.items():
            x, y = pos
            d = math.sqrt((x - self.epicenter[0])**2 + (y - self.epicenter[1])**2)
            if d < self.disaster_radius / 3:
                baseline = 5
            elif d < 2 * self.disaster_radius / 3:
                baseline = 4
            elif d < self.disaster_radius:
                baseline = 3
            else:
                baseline = 0
            diff = baseline - level
            if diff > 0:
                change = random.randint(1, int(self.disaster_dynamics))
            elif diff < 0:
                change = -random.randint(1, int(self.disaster_dynamics))
            else:
                change = 0
            new_level = level + change
            if random.random() < self.shock_probability:
                shock = random.randint(1, self.shock_magnitude)
                if random.random() < 0.5:
                    shock = -shock
                new_level += shock
            new_level = max(0, min(5, new_level))
            new_grid[pos] = new_level
        if random.random() < 0.02:
            self.epicenter = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.disaster_grid = new_grid

    def step(self):
        self.tokens_this_tick = {}  # To record tokens delivered this tick.
        self.update_disaster()
        self.schedule.step()
        # Compute unmet needs: count cells with level>=4 that got no token.
        unmet = 0
        for pos, level in self.disaster_grid.items():
            if level >= 4 and self.tokens_this_tick.get(pos, 0) == 0:
                unmet += 1
        self.unmet_needs_evolution.append(unmet)
        total_reward_exploit = 0
        total_reward_explor = 0
        for agent in self.humans.values():
            agent.process_relief_actions(self.tick, self.disaster_grid)
            if agent.agent_type == "exploitative":
                total_reward_exploit += agent.total_reward
            else:
                total_reward_explor += agent.total_reward
        exp_human_trust = []
        exp_ai_trust = []
        expl_human_trust = []
        expl_ai_trust = []
        calls_exp_human = calls_exp_ai = calls_expl_human = calls_expl_ai = 0
        for agent in self.humans.values():
            human_vals = [v for key, v in agent.trust.items() if key.startswith("H_")]
            ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
            if agent.agent_type == "exploitative":
                if human_vals:
                    exp_human_trust.append(sum(human_vals) / len(human_vals))
                if ai_vals:
                    exp_ai_trust.append(sum(ai_vals) / len(ai_vals))
                calls_exp_human += agent.calls_human
                calls_exp_ai += agent.calls_ai
            else:
                if human_vals:
                    expl_human_trust.append(sum(human_vals) / len(human_vals))
                if ai_vals:
                    expl_ai_trust.append(sum(ai_vals) / len(ai_vals))
                calls_expl_human += agent.calls_human
                calls_expl_ai += agent.calls_ai
            agent.calls_human = 0
            agent.calls_ai = 0
            agent.total_reward = 0

        avg_exp_human_trust = sum(exp_human_trust) / len(exp_human_trust) if exp_human_trust else 0
        avg_exp_ai_trust = sum(exp_ai_trust) / len(exp_ai_trust) if exp_ai_trust else 0
        avg_expl_human_trust = sum(expl_human_trust) / len(expl_human_trust) if expl_human_trust else 0
        avg_expl_ai_trust = sum(expl_ai_trust) / len(expl_ai_trust) if expl_ai_trust else 0

        self.trust_data.append((avg_exp_human_trust, avg_exp_ai_trust, avg_expl_human_trust, avg_expl_ai_trust))
        self.calls_data.append((calls_exp_human, calls_exp_ai, calls_expl_human, calls_expl_ai))
        self.rewards_data.append((total_reward_exploit, total_reward_explor))
        self.tick += 1

#########################################
# Agent Definitions
#########################################
class HumanAgent(Agent):
    def __init__(self, unique_id, model, id_num, agent_type="exploitative", share_confirming=0.5):
        super().__init__(model)
        self.unique_id = unique_id
        self.id_num = id_num
        self.model = model
        self.agent_type = agent_type
        # Attitude: "confirming" agents get D=0.3, delta=20, epsilon=5; others get D=0.5, delta=5, epsilon=3.
        if random.random() < share_confirming:
            self.attitude_type = "confirming"
            self.D = 0.3
            self.delta = 20
            self.epsilon = 5
        else:
            self.attitude_type = "other"
            self.D = 0.5
            self.delta = 5
            self.epsilon = 3

        self.trust = {}
        self.info_accuracy = {}
        self.Q = {}  # Q-values for candidate sources.
        self.beliefs = {(x, y): 0 for x in range(self.model.width) for y in range(self.model.height)}
        # pending_relief entries: (tick, source_id, accepted_count, confirmations)
        self.pending_relief = []
        self.calls_human = 0
        self.calls_ai = 0
        self.total_reward = 0
        self.learning_rate = 0.1
        self.info_mode = "human"
        # Friend list is set in the model initialization.
        self.friends = set()
        # RL parameters for information search:
        self.lambda_parameter = 0.1   # Exploration probability (ε in ε–greedy)
        self.q_parameter = 0.9        # Scaling factor for initializing Q-values

    def sense_environment(self):
        pos = self.pos
        cells = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
        for cell in cells:
            actual = self.model.disaster_grid[cell]
            if random.random() < 0.3:
                self.beliefs[cell] = max(0, min(5, actual + random.choice([-1, 1])))
            else:
                self.beliefs[cell] = actual

    def request_information(self):
        # Build candidate lists for humans and for AI.
        human_candidates = []
        ai_candidates = []
        for candidate in self.trust:
            if candidate.startswith("H_"):
                bonus = 0.1 if candidate in self.friends else 0.0
                if candidate not in self.Q:
                    # For humans, exploitative agents emphasize trust; exploratory agents weight accuracy more.
                    if self.agent_type == "exploitative":
                        self.Q[candidate] = (self.trust[candidate] + bonus) * self.q_parameter
                    else:
                        self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.7 + self.trust[candidate] * 0.3) + bonus) * self.q_parameter
                human_candidates.append((candidate, self.Q[candidate]))
            elif candidate.startswith("A_"):
                if candidate not in self.Q:
                    coverage_bonus = 1.2  # AI coverage bonus factor.
                    if self.agent_type == "exploitative":
                        self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.4 + self.trust[candidate] * 0.6)) * self.q_parameter * coverage_bonus
                    else:
                        self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.6 + self.trust[candidate] * 0.4)) * self.q_parameter * coverage_bonus
                ai_candidates.append((candidate, self.Q[candidate]))
        # Mode decision: use different multipliers by agent type.
        if self.agent_type == "exploitative":
            best_human = max([q for _, q in human_candidates]) if human_candidates else 0
            multiplier = 3.0
        else:
            best_human = max([q for _, q in human_candidates]) if human_candidates else 0
            multiplier = 1.0  # Exploratory agents are less inclined to call humans.
        best_ai = max([q for _, q in ai_candidates]) if ai_candidates else 0
        effective_human = best_human * multiplier
        if random.random() < self.lambda_parameter:
            mode_choice = random.choice(["human", "ai"])
        else:
            mode_choice = "human" if effective_human >= best_ai else "ai"

        accepted_counts = {}
        if mode_choice == "human":
            # Select up to 3 human candidates via ε–greedy selection.
            candidate_pool = human_candidates.copy()
            num_calls = min(3, len(candidate_pool))
            selected = []
            for _ in range(num_calls):
                if random.random() < self.lambda_parameter:
                    choice = random.choice(candidate_pool)
                else:
                    choice = max(candidate_pool, key=lambda x: x[1])
                selected.append(choice)
                candidate_pool.remove(choice)
            for candidate, q_val in selected:
                self.calls_human += 1
                accepted = 0
                confirmations = 0
                other = self.model.humans.get(candidate)
                if other is not None:
                    rep = other.provide_information_full()
                    # Simulate non-response if provider is in a high-destruction cell.
                    other_pos = other.pos
                    cell_level = self.model.disaster_grid[other_pos]
                    if cell_level >= 3 and random.random() < ((cell_level - 2) * 0.2):
                        rep = None
                    if rep is not None:
                        for cell, reported_value in rep.items():
                            old_belief = self.beliefs[cell]
                            d = abs(reported_value - old_belief)
                            P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                            # Update trust differently for each agent type.
                            if random.random() < P_accept:
                                self.beliefs[cell] = reported_value
                                accepted += 1
                                if reported_value == old_belief:
                                    confirmations += 1
                                # Exploitative agents get a robust boost.
                                if self.agent_type == "exploitative":
                                    self.trust[candidate] = min(1, self.trust[candidate] + 0.05)
                                else:
                                    self.trust[candidate] = min(1, self.trust[candidate] + 0.03)
                            else:
                                # Exploratory agents are more punitive.
                                if self.agent_type == "exploitative":
                                    self.trust[candidate] = max(0, self.trust[candidate] - 0.05)
                                else:
                                    self.trust[candidate] = max(0, self.trust[candidate] - 0.1)
                accepted_counts[candidate] = (accepted, confirmations)
                self.pending_relief.append((self.model.tick, candidate, accepted, confirmations))
        else:  # mode_choice == "ai"
            candidate_pool = ai_candidates.copy()
            if candidate_pool:
                if random.random() < self.lambda_parameter:
                    selected = [random.choice(candidate_pool)]
                else:
                    selected = [max(candidate_pool, key=lambda x: x[1])]
            else:
                selected = []
            for candidate, q_val in selected:
                self.calls_ai += 1
                accepted = 0
                confirmations = 0
                other = self.model.ais.get(candidate)
                if other is not None:
                    rep = other.provide_information_full(self.beliefs, trust=self.trust[candidate])
                    for cell, reported_value in rep.items():
                        old_belief = self.beliefs[cell]
                        d = abs(reported_value - old_belief)
                        P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                        if random.random() < P_accept:
                            self.beliefs[cell] = reported_value
                            accepted += 1
                            if reported_value == old_belief:
                                confirmations += 1
                            # Similar trust update for AI candidates.
                            if self.agent_type == "exploitative":
                                self.trust[candidate] = min(1, self.trust[candidate] + 0.05)
                            else:
                                self.trust[candidate] = min(1, self.trust[candidate] + 0.03)
                        else:
                            if self.agent_type == "exploitative":
                                self.trust[candidate] = max(0, self.trust[candidate] - 0.05)
                            else:
                                self.trust[candidate] = max(0, self.trust[candidate] - 0.1)
                accepted_counts[candidate] = (accepted, confirmations)
                self.pending_relief.append((self.model.tick, candidate, accepted, confirmations))

    def send_relief(self):
        pos = self.pos
        # Deliver tokens to up to 6 cells (if beliefs indicate sufficient need).
        if self.info_mode == "ai":
            cells = self.model.grid.get_neighborhood(pos, moore=True, radius=5, include_center=True)
        else:
            cells = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
        sorted_cells = sorted(cells, key=lambda c: self.beliefs[c], reverse=True)
        selected = [c for c in sorted_cells if self.beliefs[c] >= 3][:6]
        for cell in selected:
            self.pending_relief.append((self.model.tick, None, 0, 0))

    def process_relief_actions(self, current_tick, disaster_grid):
        new_pending = []
        for entry in self.pending_relief:
            t, source_id, accepted_count, confirmations = entry
            if current_tick - t >= 2:
                sample_cell = random.choice(list(self.beliefs.keys()))
                level = disaster_grid[sample_cell]
                reward = 2 if level == 4 else (5 if level == 5 else 0)
                if level >= 4 and (self.model.assistance_exploit.get(sample_cell, 0) +
                                   self.model.assistance_explor.get(sample_cell, 0)) == 0:
                    reward = 10
                if level <= 2:
                    reward = -0.05 * accepted_count
                self.total_reward += reward
                if level >= 4:
                    if self.agent_type == "exploitative":
                        self.model.assistance_exploit[sample_cell] = self.model.assistance_exploit.get(sample_cell, 0) + 1
                    else:
                        self.model.assistance_explor[sample_cell] = self.model.assistance_explor.get(sample_cell, 0) + 1
                    self.model.tokens_this_tick[sample_cell] = self.model.tokens_this_tick.get(sample_cell, 0) + 1
                if source_id is not None and accepted_count > 0:
                    if self.agent_type == "exploitative":
                        # For exploitative agents, confirmation is the key.
                        confirmation_ratio = confirmations / accepted_count
                        correction = 0.1 * (reward / (10 * accepted_count))
                        Q_target = confirmation_ratio + correction
                    else:
                        # For exploratory agents, accuracy (reward) is the target.
                        Q_target = reward / (10 * accepted_count)
                    old_Q = self.Q.get(source_id, self.trust.get(source_id, 0.5))
                    if source_id.startswith("H_"):
                        self.Q[source_id] = old_Q + self.learning_rate * (Q_target - old_Q)
                    else:
                        coverage_bonus = 1.2
                        self.Q[source_id] = old_Q + self.learning_rate * (Q_target * coverage_bonus - old_Q)
                    old_acc = self.info_accuracy.get(source_id, 0.5)
                    new_acc = old_acc + self.learning_rate * (Q_target - old_acc)
                    self.info_accuracy[source_id] = max(0, min(1, new_acc))
            else:
                new_pending.append(entry)
        self.pending_relief = new_pending

    def provide_information_full(self):
        pos = self.pos
        cells = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
        info = {}
        for cell in cells:
            level = self.model.disaster_grid[cell]
            if random.random() < 0.1:
                level = max(0, min(5, level + random.choice([-1, 1])))
            info[cell] = level
        return info

    def step(self):
        self.sense_environment()
        self.request_information()
        self.send_relief()

class AIAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.model = model
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
            if abs(sensed_val - human_val) > 1:
                correction_factor = 1 - min(1, trust)
                corrected = round(sensed_val + correction_factor * (human_val - sensed_val))
            else:
                corrected = sensed_val
            info[cell] = corrected
        return info

    def step(self):
        self.sense_environment()

#########################################
# Main: Run Simulation and Output Visuals
#########################################
if __name__ == "__main__":
    share_exploitative = 0.5
    share_of_disaster = 0.2
    initial_trust = 0.5
    initial_ai_trust = 0.5
    number_of_humans = 50
    share_confirming = 0.5  # 50% confirming agents.
    disaster_dynamics = 2
    shock_probability = 0.1
    shock_magnitude = 2
    trust_update_mode = "average"
    exploitative_correction_factor = 1.0
    width = 50
    height = 50

    model = DisasterModel(share_exploitative, share_of_disaster, initial_trust, initial_ai_trust,
                          number_of_humans, share_confirming, disaster_dynamics, shock_probability, shock_magnitude,
                          trust_update_mode, exploitative_correction_factor, width, height)
    ticks = 600
    for i in range(ticks):
        model.step()

    # Visual 1: Histogram of tokens delivered to cells in need (by agent type).
    tokens_exploit = [model.assistance_exploit.get(pos, 0) for pos, level in model.disaster_grid.items() if level >= 4]
    tokens_explor = [model.assistance_explor.get(pos, 0) for pos, level in model.disaster_grid.items() if level >= 4]
    plt.figure()
    plt.hist([tokens_exploit, tokens_explor],
             bins=range(0, max(max(tokens_exploit, default=0), max(tokens_explor, default=0)) + 2),
             label=["Exploitative", "Exploratory"],
             color=["skyblue", "lightgreen"], edgecolor='black')
    plt.title("Histogram: Assistance Tokens Delivered\n(to Cells in Need, Level 4 or 5)")
    plt.xlabel("Total Tokens Delivered")
    plt.ylabel("Number of Cells")
    plt.legend()
    plt.show()

    # Visual 2: Time series of unmet needs.
    plt.figure()
    plt.plot(range(len(model.unmet_needs_evolution)), model.unmet_needs_evolution, marker='o')
    plt.title("Time Series: Unmet Needs\n(Number of Cells in Need Without Assistance)")
    plt.xlabel("Tick")
    plt.ylabel("Unassisted Cells (Level ≥ 4)")
    plt.show()

    # Visual 3: Trust evolution by agent type.
    ticks_range = list(range(ticks))
    exp_human_trust = [d[0] for d in model.trust_data]
    exp_ai_trust = [d[1] for d in model.trust_data]
    expl_human_trust = [d[2] for d in model.trust_data]
    expl_ai_trust = [d[3] for d in model.trust_data]
    plt.figure()
    plt.plot(ticks_range, exp_human_trust, label="Exploitative: Human Trust")
    plt.plot(ticks_range, exp_ai_trust, label="Exploitative: AI Trust")
    plt.plot(ticks_range, expl_human_trust, label="Exploratory: Human Trust")
    plt.plot(ticks_range, expl_ai_trust, label="Exploratory: AI Trust")
    plt.xlabel("Tick")
    plt.ylabel("Average Trust")
    plt.title("Trust Evolution by Agent Type")
    plt.legend()
    plt.show()

    # Visual 4: Information request calls by agent type.
    calls_exp_human = [d[0] for d in model.calls_data]
    calls_exp_ai = [d[1] for d in model.calls_data]
    calls_expl_human = [d[2] for d in model.calls_data]
    calls_expl_ai = [d[3] for d in model.calls_data]
    plt.figure()
    plt.plot(ticks_range, calls_exp_human, label="Exploitative: Calls to Humans")
    plt.plot(ticks_range, calls_exp_ai, label="Exploitative: Calls to AI")
    plt.plot(ticks_range, calls_expl_human, label="Exploratory: Calls to Humans")
    plt.plot(ticks_range, calls_expl_ai, label="Exploratory: Calls to AI")
    plt.xlabel("Tick")
    plt.ylabel("Information Requests")
    plt.title("Information Request Calls by Agent Type")
    plt.legend()
    plt.show()
