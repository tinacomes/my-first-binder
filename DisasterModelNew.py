#!/usr/bin/env python
import random
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
                 ai_alignment_level=0.5,      # ai alignment
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
        self.ai_alignment_level = ai_alignment_level

        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.tick = 0

        # For tracking assistance:
        self.assistance_exploit = {}  # cell -> tokens delivered by exploitative agents (correct assistance)
        self.assistance_explor = {}   # cell -> tokens delivered by exploratory agents (correct assistance)
        self.assistance_incorrect_exploit = {}  # for cells with level <= 2
        self.assistance_incorrect_explor = {}

        self.unmet_needs_evolution = []  # Number of cells in need (level>=4) that got no token this tick.

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

        # Create a Watts–Strogatz network for friend selection.
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

        # Initialize trust and info_accuracy for each human.
        for i in range(self.num_humans):
            agent_id = f"H_{i}"
            agent = self.humans[agent_id]
            for j in range(self.num_humans):
                if agent_id == f"H_{j}":
                    continue
                agent.trust[f"H_{j}"] = random.uniform(self.base_trust - 0.05, self.base_trust + 0.05)
                agent.info_accuracy[f"H_{j}"] = random.uniform(0.4, 0.6)
            # Set friend list.
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

        # Data tracking.
        self.trust_data = []   # Tuple per tick: (exp_human, exp_ai, expl_human, expl_ai)
        self.calls_data = []   # Tuple per tick: (calls_exp_human, calls_exp_ai, calls_expl_human, calls_expl_ai)
        self.rewards_data = [] # Tuple per tick: (total_reward_exploitative, total_reward_exploratory)

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
        self.tokens_this_tick = {}  # Reset tokens this tick.
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
                    exp_human_trust.append(np.mean(human_vals))
                if ai_vals:
                    exp_ai_trust.append(np.mean(ai_vals))
                calls_exp_human += agent.calls_human
                calls_exp_ai += agent.calls_ai
            else:
                if human_vals:
                    expl_human_trust.append(np.mean(human_vals))
                if ai_vals:
                    expl_ai_trust.append(np.mean(ai_vals))
                calls_expl_human += agent.calls_human
                calls_expl_ai += agent.calls_ai
            agent.calls_human = 0
            agent.calls_ai = 0
            agent.total_reward = 0

        avg_exp_human_trust = np.mean(exp_human_trust) if exp_human_trust else 0
        avg_exp_ai_trust = np.mean(exp_ai_trust) if exp_ai_trust else 0
        avg_expl_human_trust = np.mean(expl_human_trust) if expl_human_trust else 0
        avg_expl_ai_trust = np.mean(expl_ai_trust) if expl_ai_trust else 0

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
        # Attitude parameters.
        if random.random() < share_confirming:
            self.attitude_type = "confirming"
            self.D = 0.3
            self.delta = 20
            self.epsilon = 5
        else:
            self.attitude_type = "other"
            self.D = 0.9
            self.delta = 3
            self.epsilon = 3

        self.trust = {}
        self.info_accuracy = {}
        self.Q = {}  # Q-values for candidate sources.
        self.beliefs = {(x, y): 0 for x in range(self.model.width) for y in range(self.model.height)}
        self.pending_relief = []  # Each entry: (tick, source_id, accepted_count, confirmations, target_cell)
        self.calls_human = 0
        self.calls_ai = 0
        self.total_reward = 0
        self.learning_rate = 0.1
        self.info_mode = "human"
        self.friends = set()  # Set in model initialization.
        self.lambda_parameter = 0.1   # Exploration probability.
        self.q_parameter = 0.9        # Scaling factor for Q-values.
        self.delayed_reports = []  # For exploratory agents.
        self.ai_reported = {}      # To record AI-provided information.

    def sense_environment(self):
        pos = self.pos
        radius = 1 if self.agent_type == "exploitative" else 5
        cells = self.model.grid.get_neighborhood(pos, moore=True, radius=radius, include_center=True)
        for cell in cells:
            actual = self.model.disaster_grid[cell]
            if random.random() < 0.3:
                self.beliefs[cell] = max(0, min(5, actual + random.choice([-1, 1])))
            else:
                self.beliefs[cell] = actual

    def request_information(self):
        human_candidates = []
        ai_candidates = []
        if self.agent_type == "exploitative":
            all_human_candidates = []
            local_human_candidates = []
            all_ai_candidates = []
            local_ai_candidates = []
            for candidate in self.trust:
                if candidate.startswith("H_"):
                    bonus = 0.1 if candidate in self.friends else 0.0
                    if candidate not in self.Q:
                        self.Q[candidate] = (self.trust[candidate] + bonus) * self.q_parameter
                    candidate_tuple = (candidate, self.Q[candidate])
                    all_human_candidates.append(candidate_tuple)
                    candidate_agent = self.model.humans.get(candidate)
                    if candidate_agent is not None:
                        dist = math.sqrt((self.pos[0] - candidate_agent.pos[0])**2 +
                                         (self.pos[1] - candidate_agent.pos[1])**2)
                        if dist <= 2:
                            local_human_candidates.append(candidate_tuple)
                elif candidate.startswith("A_"):
                    if candidate not in self.Q:
                        coverage_bonus = 1.2
                        self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.4 + self.trust[candidate] * 0.6)) * self.q_parameter * coverage_bonus
                    candidate_tuple = (candidate, self.Q[candidate])
                    all_ai_candidates.append(candidate_tuple)
                    candidate_agent = self.model.ais.get(candidate)
                    if candidate_agent is not None and hasattr(candidate_agent, 'pos'):
                        try:
                            dist = math.sqrt((self.pos[0] - candidate_agent.pos[0])**2 +
                                             (self.pos[1] - candidate_agent.pos[1])**2)
                            if dist <= 2:
                                local_ai_candidates.append(candidate_tuple)
                        except Exception:
                            local_ai_candidates.append(candidate_tuple)
            human_candidates = local_human_candidates if local_human_candidates else all_human_candidates
            ai_candidates = local_ai_candidates if local_ai_candidates else all_ai_candidates
        else:
            for candidate in self.trust:
                if candidate.startswith("H_"):
                    bonus = 0.1 if candidate in self.friends else 0.0
                    if candidate not in self.Q:
                        self.Q[candidate] = (((self.info_accuracy.get(candidate, 0.5) * 0.7) + (self.trust[candidate] * 0.3)) + bonus) * self.q_parameter
                    human_candidates.append((candidate, self.Q[candidate]))
                elif candidate.startswith("A_"):
                    if candidate not in self.Q:
                        coverage_bonus = 1.2
                        self.Q[candidate] = ((self.info_accuracy.get(candidate, 0.5) * 0.6 + self.trust[candidate] * 0.4)) * self.q_parameter * coverage_bonus
                    ai_candidates.append((candidate, self.Q[candidate]))
        if self.agent_type == "exploitative":
            best_human = max([q for _, q in human_candidates]) if human_candidates else 0
            multiplier = 3.0
        else:
            best_human = max([q for _, q in human_candidates]) if human_candidates else 0
            multiplier = 1.0
        best_ai = max([q for _, q in ai_candidates]) if ai_candidates else 0
        effective_human = best_human * multiplier
        mode_choice = "human" if (random.random() >= self.lambda_parameter and effective_human >= best_ai) else "ai"
        aggregated_reports = {}
        accepted_counts = {}
        if mode_choice == "human":
            candidate_pool = human_candidates.copy()
            num_calls = 3  # Force 3 calls.
            selected = []
            for _ in range(num_calls):
                if candidate_pool:
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
                    other_pos = other.pos
                    cell_level = self.model.disaster_grid[other_pos]
                    if cell_level >= 3 and random.random() < ((cell_level - 2) * 0.2):
                        rep = None
                    if rep is not None:
                        for cell, reported_value in rep.items():
                            aggregated_reports.setdefault(cell, []).append(reported_value)
                            old_belief = self.beliefs[cell]
                            d = abs(reported_value - old_belief)
                            P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                            if random.random() < P_accept:
                                accepted += 1
                                if reported_value == old_belief:
                                    confirmations += 1
                                self.trust[candidate] = min(1, self.trust[candidate] + 0.05)
                            else:
                                self.trust[candidate] = max(0, self.trust[candidate] - 0.05)
                accepted_counts[candidate] = (accepted, confirmations)
                self.pending_relief.append((self.model.tick, candidate, accepted, confirmations))
        else:  # mode_choice == "ai"
            candidate_pool = ai_candidates.copy()
            if candidate_pool:
                selected = [max(candidate_pool, key=lambda x: x[1])] if random.random() >= self.lambda_parameter else [random.choice(candidate_pool)]
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
                        aggregated_reports.setdefault(cell, []).append(reported_value)
                        old_belief = self.beliefs[cell]
                        d = abs(reported_value - old_belief)
                        P_accept = 1.0 if d == 0 else (self.D ** self.delta) / ((d ** self.delta) + (self.D ** self.delta))
                        if random.random() < P_accept:
                            accepted += 1
                            if reported_value == old_belief:
                                confirmations += 1
                            self.trust[candidate] = min(1, self.trust[candidate] + 0.03)
                        else:
                            self.trust[candidate] = max(0, self.trust[candidate] - 0.1)
                accepted_counts[candidate] = (accepted, confirmations)
                self.pending_relief.append((self.model.tick, candidate, accepted, confirmations))
        # Record AI reports if AI mode was used.
        if mode_choice == "ai":
            for cell, rep_val in aggregated_reports.items():
                self.ai_reported.setdefault(cell, []).append(rep_val)
        # Update beliefs.
        if self.agent_type == "exploitative":
            for cell, reports in aggregated_reports.items():
                avg_report = sum(reports) / len(reports)
                current_value = self.beliefs[cell]
                difference = avg_report - current_value
                scaling = 1 + 0.1 * (len(reports) - 1)
                self.beliefs[cell] = max(0, min(5, current_value + self.learning_rate * scaling * difference))
        else:
            if aggregated_reports:
                self.delayed_reports.append((self.model.tick, aggregated_reports))
                
    def update_delayed_beliefs(self):
        new_buffer = []
        for t, reports in self.delayed_reports:
            if self.model.tick - t >= 2:
                for cell, rep_list in reports.items():
                    avg_report = sum(rep_list) / len(rep_list)
                    current_value = self.beliefs[cell]
                    difference = avg_report - current_value
                    scaling = 1 + 0.1 * (len(rep_list) - 1)
                    self.beliefs[cell] = max(0, min(5, current_value + self.learning_rate * scaling * difference))
            else:
                new_buffer.append((t, reports))
        self.delayed_reports = new_buffer

    def send_relief(self):
        tokens_to_send = 5
        if self.agent_type == "exploitative":
            cells = self.model.grid.get_neighborhood(self.pos, moore=True, radius=1, include_center=True)
        else:
            cells = list(self.model.disaster_grid.keys())
        friend_positions = set()
        for friend_id in self.friends:
            if friend_id in self.model.humans:
                friend_positions.add(self.model.humans[friend_id].pos)
        def cell_score(cell):
            score = self.beliefs.get(cell, 0)
            if cell in friend_positions:
                score += 1
            return score
        sorted_cells = sorted(cells, key=cell_score, reverse=True)
        selected = [c for c in sorted_cells if cell_score(c) >= 3][:tokens_to_send]
        for cell in selected:
            self.pending_relief.append((self.model.tick, None, 0, 0, cell))

    def process_relief_actions(self, current_tick, disaster_grid):
        new_pending = []
        for entry in self.pending_relief:
            if len(entry) == 5:
                t, source_id, accepted_count, confirmations, target_cell = entry
            else:
                t, source_id, accepted_count, confirmations = entry
                target_cell = random.choice(list(self.beliefs.keys()))
            if current_tick - t >= 2:
                level = disaster_grid[target_cell]
                reward = 2 if level == 4 else (5 if level == 5 else 0)
                if level >= 4 and (self.model.assistance_exploit.get(target_cell, 0) + self.model.assistance_explor.get(target_cell, 0)) == 0:
                    reward = 10
                if level <= 2:
                    reward = -0.05 * accepted_count
                self.total_reward += reward
                if level >= 4:
                    if self.agent_type == "exploitative":
                        self.model.assistance_exploit[target_cell] = self.model.assistance_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_explor[target_cell] = self.model.assistance_explor.get(target_cell, 0) + 1
                    self.model.tokens_this_tick[target_cell] = self.model.tokens_this_tick.get(target_cell, 0) + 1
                elif level <= 2:
                    if self.agent_type == "exploitative":
                        self.model.assistance_incorrect_exploit[target_cell] = self.model.assistance_incorrect_exploit.get(target_cell, 0) + 1
                    else:
                        self.model.assistance_incorrect_explor[target_cell] = self.model.assistance_incorrect_explor.get(target_cell, 0) + 1
                    self.model.tokens_this_tick[target_cell] = self.model.tokens_this_tick.get(target_cell, 0) + 1
                if source_id is not None and accepted_count > 0:
                    if self.agent_type == "exploitative":
                        confirmation_ratio = confirmations / accepted_count
                        Q_target = 2 * confirmation_ratio  # Strong reinforcement for confirmation.
                    else:
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
        if self.agent_type == "exploratory":
            self.update_delayed_beliefs()

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
                # Use the model's alignment parameter instead of trust directly
                alignment_factor = self.model.ai_alignment_level * (1 - min(1, trust))
                corrected = round(sensed_val + alignment_factor * (human_val - sensed_val))
            else:
                corrected = sensed_val
            info[cell] = corrected
        return info
        
    def step(self):
        self.sense_environment()

#########################################
# Main: Run Simulation and Generate Outputs for Validation
#########################################
if __name__ == "__main__":
    share_exploitative = 0.5
    share_of_disaster = 0.2
    initial_trust = 0.5
    initial_ai_trust = 0.75
    number_of_humans = 50
    share_confirming = 0.5  # 50% confirming agents.
    disaster_dynamics = 2
    shock_probability = 0.1
    shock_magnitude = 2
    trust_update_mode = "average"
    exploitative_correction_factor = 1.0
    width = 50
    height = 50

    # Run the simulation for a number of ticks.
    ticks = 600
    model = DisasterModel(share_exploitative, share_of_disaster, initial_trust, initial_ai_trust,
                          number_of_humans, share_confirming, disaster_dynamics, shock_probability, shock_magnitude,
                          trust_update_mode, exploitative_correction_factor, width, height)
    for i in range(ticks):
        model.step()

    # Visual 1: Histogram of tokens delivered to cells in need (level >= 4) by agent type.
    tokens_exploit = [model.assistance_exploit.get(pos, 0) for pos, level in model.disaster_grid.items() if level >= 4]
    tokens_explor = [model.assistance_explor.get(pos, 0) for pos, level in model.disaster_grid.items() if level >= 4]
    max_tokens = max(max(tokens_exploit, default=0), max(tokens_explor, default=0))
    bin_width = 50
    if max_tokens < bin_width:
        bins_correct = [0, bin_width]
    else:
        bins_correct = list(range(0, max_tokens + bin_width, bin_width))
    plt.figure()
    plt.hist([tokens_exploit, tokens_explor],
             bins=bins_correct,
             label=["Exploitative", "Exploratory"],
             color=["skyblue", "lightgreen"],
             edgecolor='black')
    plt.title("Histogram: Assistance Tokens Delivered\n(to Cells in Need, Level 4 or 5)")
    plt.xlabel("Total Tokens Delivered")
    plt.ylabel("Number of Cells")
    plt.legend()
    plt.show()

    # Visual 2: Histogram of tokens incorrectly delivered (cells with level <= 2).
    tokens_incorrect_exploit = [model.assistance_incorrect_exploit.get(pos, 0) for pos, level in model.disaster_grid.items() if level <= 2]
    tokens_incorrect_explor = [model.assistance_incorrect_explor.get(pos, 0) for pos, level in model.disaster_grid.items() if level <= 2]
    max_tokens_incorrect = max(max(tokens_incorrect_exploit, default=0), max(tokens_incorrect_explor, default=0))
    if max_tokens_incorrect < bin_width:
        bins_incorrect = [0, bin_width]
    else:
        bins_incorrect = list(range(0, max_tokens_incorrect + bin_width, bin_width))
    plt.figure()
    plt.hist([tokens_incorrect_exploit, tokens_incorrect_explor],
             bins=bins_incorrect,
             label=["Exploitative (Incorrect)", "Exploratory (Incorrect)"],
             color=["coral", "orchid"],
             edgecolor='black')
    plt.title("Histogram: Incorrect Assistance Tokens Delivered\n(to Cells with Level 2 or Lower)")
    plt.xlabel("Total Tokens Delivered")
    plt.ylabel("Number of Cells")
    plt.legend()
    plt.show()

    # Visual 3: Time series of unmet needs.
    plt.figure()
    plt.plot(range(len(model.unmet_needs_evolution)), model.unmet_needs_evolution, marker='o')
    plt.title("Time Series: Unmet Needs\n(Number of Cells in Need Without Assistance)")
    plt.xlabel("Tick")
    plt.ylabel("Unassisted Cells (Level ≥ 4)")
    plt.show()

    # Visual 4: Trust evolution by agent type.
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

    # Visual 5: Information request calls by agent type.
    calls_exp_human = [d[0]/5 for d in model.calls_data]  # dividing by 5 for scaling, if needed.
    calls_exp_ai = [d[1] for d in model.calls_data]
    calls_expl_human = [d[2]/5 for d in model.calls_data]
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
