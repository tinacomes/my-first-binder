import random, math
import matplotlib.pyplot as plt
import networkx as nx

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.agent import Agent as MesaAgent


#############################
# Model Definition
#############################
class DisasterModel(Model):
    def __init__(self, share_exploitative, share_of_disaster, initial_trust, number_of_humans, width=50, height=50):
        """
        Parameters:
         - share_exploitative: Fraction of human agents using an exploitative strategy.
         - share_of_disaster: Fraction of grid cells affected (used to compute disaster radius).
         - initial_trust: Baseline trust value (will be used to derive trust levels).
         - number_of_humans: Total number of human agents.
         
        Fixed number of AI agents is 5.
        """
        super().__init__()  # Initialize the base Mesa Model (creates self.random, etc.)
        self.share_exploitative = share_exploitative
        self.share_of_disaster = share_of_disaster
        self.base_trust = initial_trust  # this value is used as a baseline.
        self.num_humans = number_of_humans
        self.num_ai = 5
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.tick = 0

        # --- Create disaster grid (each cell gets a destruction level 0 to 5) ---
        # We choose an epicenter at random and compute a radius so that approximately
        # share_of_disaster fraction of cells are affected.
        self.disaster_grid = {}
        self.epicenter = (random.randint(0, width - 1), random.randint(0, height - 1))
        # Compute radius so that: area = πr² ≈ share_of_disaster*total_cells.
        self.disaster_radius = math.sqrt(self.share_of_disaster * width * height / math.pi)
        for x in range(width):
            for y in range(height):
                d = math.sqrt((x - self.epicenter[0]) ** 2 + (y - self.epicenter[1]) ** 2)
                if d < self.disaster_radius / 3:
                    level = 5
                elif d < 2 * self.disaster_radius / 3:
                    level = 4
                elif d < self.disaster_radius:
                    level = 3
                else:
                    level = 0
                self.disaster_grid[(x, y)] = level

        # --- Create social network among human agents ---
        # We use a Watts–Strogatz small-world network.
        self.social_network = nx.watts_strogatz_graph(self.num_humans, 4, 0.1)

        # --- Create Human agents ---
        self.humans = {}
        for i in range(self.num_humans):
            # Determine strategy by share_exploitative probability.
            agent_type = "exploitative" if random.random() < self.share_exploitative else "exploratory"
            a = HumanAgent(unique_id=f"H_{i}", model=self, agent_type=agent_type)
            self.humans[f"H_{i}"] = a
            self.schedule.add(a)
            # Place the agent at a random grid cell.
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(a, (x, y))

        # --- Initialize trust for human agents ---
        # For every human, for every other human: if connected in the social network, set a higher trust;
        # for AI agents, use a lower value.
        for i in range(self.num_humans):
            agent_id = f"H_{i}"
            agent = self.humans[agent_id]
            for j in range(self.num_humans):
                if agent_id == f"H_{j}":
                    continue
                if self.social_network.has_edge(i, j):
                    # Neighbour in the network gets a “bonus”
                    agent.trust[f"H_{j}"] = self.base_trust + 0.1  
                else:
                    agent.trust[f"H_{j}"] = self.base_trust
            # For AI agents:
            for k in range(self.num_ai):
                agent.trust[f"A_{k}"] = max(0, self.base_trust - 0.1)

        # --- Create AI agents ---
        self.ais = {}
        for k in range(self.num_ai):
            a = AIAgent(unique_id=f"A_{k}", model=self)
            self.ais[f"A_{k}"] = a
            self.schedule.add(a)
            x = random.randrange(width)
            y = random.randrange(height)
            self.grid.place_agent(a, (x, y))

        # --- Statistics containers ---
        # Each tick we record:
        #   trust_data: tuple (avg human trust, avg AI trust) for exploitative and exploratory agents
        #   calls_data: counts of info requests (to humans and AI) for both strategies
        #   rewards_data: total rewards received by humans in the tick.
        self.trust_data = []   # each entry: (exp_human, exp_ai, expl_human, expl_ai)
        self.calls_data = []   # each entry: (calls_human_exp, calls_ai_exp, calls_human_expl, calls_ai_expl)
        self.rewards_data = []  # total rewards per tick

    def update_disaster(self):
        """Update the disaster grid.
           For each cell, adjust its destruction level toward the “baseline” (computed from epicenter distance)
           and add a random shock with small probability."""
        new_grid = {}
        for pos, level in self.disaster_grid.items():
            x, y = pos
            d = math.sqrt((x - self.epicenter[0]) ** 2 + (y - self.epicenter[1]) ** 2)
            # Compute baseline destruction level (as during initialization)
            if d < self.disaster_radius / 3:
                baseline = 5
            elif d < 2 * self.disaster_radius / 3:
                baseline = 4
            elif d < self.disaster_radius:
                baseline = 3
            else:
                baseline = 0
            # Adjust level gradually toward baseline.
            if level < baseline:
                new_level = level + 1
            elif level > baseline:
                new_level = level - 1
            else:
                new_level = level
            # Add a random shock (5% chance): either increase or decrease the level.
            if random.random() < 0.05:
                if random.random() < 0.5:
                    new_level = min(5, new_level + 1)
                else:
                    new_level = max(0, new_level - 1)
            new_grid[pos] = new_level
        self.disaster_grid = new_grid

    def step(self):
        # Update the disaster field.
        self.update_disaster()
        # Let each agent act.
        self.schedule.step()

        # --- Process delayed rewards for relief actions ---
        for agent in self.humans.values():
            agent.process_relief_actions(self.tick, self.disaster_grid)

        # --- Collect statistics ---
        exp_human_trust = []
        exp_ai_trust = []
        expl_human_trust = []
        expl_ai_trust = []
        calls_human_exploit = 0
        calls_ai_exploit = 0
        calls_human_explor = 0
        calls_ai_explor = 0
        total_rewards = 0

        for agent in self.humans.values():
            if agent.agent_type == "exploitative":
                human_vals = [v for key, v in agent.trust.items() if key.startswith("H_")]
                ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
                if human_vals:
                    exp_human_trust.append(sum(human_vals) / len(human_vals))
                if ai_vals:
                    exp_ai_trust.append(sum(ai_vals) / len(ai_vals))
                calls_human_exploit += agent.calls_human
                calls_ai_exploit += agent.calls_ai
            else:
                human_vals = [v for key, v in agent.trust.items() if key.startswith("H_")]
                ai_vals = [v for key, v in agent.trust.items() if key.startswith("A_")]
                if human_vals:
                    expl_human_trust.append(sum(human_vals) / len(human_vals))
                if ai_vals:
                    expl_ai_trust.append(sum(ai_vals) / len(ai_vals))
                calls_human_explor += agent.calls_human
                calls_ai_explor += agent.calls_ai
            total_rewards += agent.total_reward
            # Reset per–tick counters.
            agent.calls_human = 0
            agent.calls_ai = 0
            agent.total_reward = 0

        avg_exp_human_trust = sum(exp_human_trust) / len(exp_human_trust) if exp_human_trust else 0
        avg_exp_ai_trust = sum(exp_ai_trust) / len(exp_ai_trust) if exp_ai_trust else 0
        avg_expl_human_trust = sum(expl_human_trust) / len(expl_human_trust) if expl_human_trust else 0
        avg_expl_ai_trust = sum(expl_ai_trust) / len(expl_ai_trust) if expl_ai_trust else 0

        self.trust_data.append((avg_exp_human_trust, avg_exp_ai_trust, avg_expl_human_trust, avg_expl_ai_trust))
        self.calls_data.append((calls_human_exploit, calls_ai_exploit, calls_human_explor, calls_ai_explor))
        self.rewards_data.append(total_rewards)

        self.tick += 1


#############################
# Agent Definitions
#############################
class HumanAgent(MesaAgent):
    def __init__(self, unique_id, model, agent_type="exploitative"):
        MesaAgent.__init__(self, model)
        # Then explicitly set the unique_id.
        # Then set the required attributes.
        self.unique_id = unique_id
        self.model = model
        self.agent_type = agent_type  # "exploitative" or "exploratory"
        self.trust = {}  # trust values for other agents; keys like "H_3" or "A_1"
        self.Q = {}      # Q–values for information requests (for simplicity, used to adjust trust updates)
        # beliefs about each grid cell; here we initialize to 0 everywhere.
        self.beliefs = {(x, y): 0 for x in range(self.model.width) for y in range(self.model.height)}
        # Pending relief actions: each item is a tuple (tick_requested, cell_position, info_provider_id)
        self.pending_relief = []
        # Counters for number of calls made in a tick.
        self.calls_human = 0
        self.calls_ai = 0
        # Total reward received in the tick.
        self.total_reward = 0
        self.learning_rate = 0.1

    def sense_environment(self):
        """Sense the destruction level on the cell the agent is on and its Moore neighborhood.
           The sensed value is assumed to be exact (i.e. without noise)."""
        pos = self.pos
        cells = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
        for cell in cells:
            self.beliefs[cell] = self.model.disaster_grid[cell]

    def request_information(self):
        """Each tick the human agent decides to request information from either two humans OR one AI.
           It chooses the mode based on comparing the highest trust values in each group."""
        # Determine the highest trust among humans and among AIs.
        human_candidates = [aid for aid in self.trust if aid.startswith("H_")]
        ai_candidates = [aid for aid in self.trust if aid.startswith("A_")]
        best_human_trust = max([self.trust[a] for a in human_candidates], default=0)
        best_ai_trust = max([self.trust[a] for a in ai_candidates], default=0)
        
        # Choose the mode based on which group has higher maximum trust.
        if best_human_trust >= best_ai_trust:
            mode = "human"
        else:
            mode = "ai"
        self.info_mode = mode  # store the mode for later use (in send_relief)
        
        responses = {}
        if mode == "human":
            # --- Request from 2 human partners ---
            if self.agent_type == "exploitative":
                sorted_humans = sorted(human_candidates, key=lambda a: self.trust[a], reverse=True)
                selected = sorted_humans[:2] if len(sorted_humans) >= 2 else sorted_humans
            else:
                selected = random.sample(human_candidates, min(2, len(human_candidates)))
            for agent_id in selected:
                self.calls_human += 1
                other = self.model.humans.get(agent_id, None)
                if other is not None:
                    resp = other.provide_information(self.pos)
                    other_pos = other.pos
                    cell_level = self.model.disaster_grid[other_pos]
                    if cell_level >= 3:
                        # At higher destruction levels, increase chance of no response.
                        prob_no_response = (cell_level - 2) * 0.2
                        if random.random() < prob_no_response:
                            resp = None
                    responses[agent_id] = resp
                else:
                    responses[agent_id] = None
        else:
            # --- Request from 1 AI partner ---
            if self.agent_type == "exploitative":
                selected_ai = max(ai_candidates, key=lambda a: self.trust[a]) if ai_candidates else None
            else:
                selected_ai = random.choice(ai_candidates) if ai_candidates else None
            if selected_ai is not None:
                self.calls_ai += 1
                other = self.model.ais.get(selected_ai, None)
                if other is not None:
                    resp = other.provide_information(self.pos, self.beliefs[self.pos])
                    responses[selected_ai] = resp
                else:
                    responses[selected_ai] = None

        # --- Update trust based on responses ---
        for agent_id, info in responses.items():
            if info is None:
                self.trust[agent_id] = max(0, self.trust[agent_id] - 0.1)
            else:
                actual = self.model.disaster_grid[self.pos]
                if abs(info - actual) == 0:
                    delta = 0.1
                elif abs(info - actual) == 1:
                    delta = 0.05
                else:
                    delta = -0.1
                self.trust[agent_id] = max(0, min(1, self.trust[agent_id] + delta))
                # Simple Q–value update.
                old_q = self.Q.get(agent_id, self.trust[agent_id])
                self.Q[agent_id] = old_q + self.learning_rate * (delta - old_q)
                self.pending_relief.append((self.model.tick, self.pos, agent_id))
    
    def send_relief(self):
        """Decide where to send relief tokens based on the current mode.
           If the agent used human information, it considers only nearby cells.
           If using AI information, it considers a larger neighborhood."""
        pos = self.pos
        # If using AI info, use a larger radius to capture distant but affected areas.
        if getattr(self, 'info_mode', 'human') == 'ai':
            cells = self.model.grid.get_neighborhood(pos, moore=True, radius=5, include_center=True)
        else:
            cells = self.model.grid.get_neighborhood(pos, moore=True, include_center=True)
        # Choose up to 3 cells with highest believed damage.
        sorted_cells = sorted(cells, key=lambda c: self.beliefs[c], reverse=True)
        selected = [c for c in sorted_cells if self.beliefs[c] >= 3][:3]
        for cell in selected:
            self.pending_relief.append((self.model.tick, cell, None))

    def process_relief_actions(self, current_tick, disaster_grid):
        """Check if any pending relief actions are 2 ticks old; if so, evaluate the reward based on the actual destruction level.
           Reward = 2 if level 4; 5 if level 5. Optionally, update trust for an info provider."""
        new_pending = []
        for t, cell, info_provider in self.pending_relief:
            if current_tick - t >= 2:
                level = disaster_grid[cell]
                reward = 0
                if level == 4:
                    reward = 2
                elif level == 5:
                    reward = 5
                self.total_reward += reward
                # For exploitative agents, update trust based on the reward.
                if info_provider is not None and self.agent_type == "exploratory":
                    if reward >= 5:
                        self.trust[info_provider] = min(1, self.trust[info_provider] + 0.05)
                    elif reward >= 2:
                        self.trust[info_provider] = min(1, self.trust[info_provider] + 0)
                    else:
                        self.trust[info_provider] = max(0, self.trust[info_provider] - 0.2)
            else:
                new_pending.append((t, cell, info_provider))
        self.pending_relief = new_pending

    def provide_information(self, pos_request):
        """When another human requests information, provide the destruction level of the requested cell.
           With 20% chance, return a noisy (deviated) value with up to 2 difference."""
        true_value = self.model.disaster_grid[pos_request]
        if random.random() < 0.2:
            deviation = random.choice([-2, 0])
            return max(0, min(5, true_value + deviation))
        else:
            return true_value

    def step(self):
        # Sense the local environment.
        self.sense_environment()
        # Request information from partners.
        self.request_information()
        # Decide and record relief actions.
        self.send_relief()


class AIAgent(MesaAgent):
    def __init__(self, unique_id, model):
        MesaAgent.__init__(self, model)
        self.unique_id = unique_id
        self.model = model
        self.memory = {}  # Can be used to remember past interactions.
        self.sensed = {}  # Store sensed values for 20% of grid cells.

    def sense_environment(self):
        """Sense a random 20% of the grid cells and store their actual destruction level."""
        num_cells = int(0.2 * self.model.width * self.model.height)
        cells = random.sample(list(self.model.disaster_grid.keys()), num_cells)
        for cell in cells:
            self.sensed[cell] = self.model.disaster_grid[cell]

    def provide_information(self, pos_request, human_belief=None):
        """Provide information about the requested cell.
           If the cell was sensed, use that value; otherwise fall back to the global grid.
           Then, if a human belief is provided and there is a large difference, correct the info towards the human's belief."""
        if pos_request in self.sensed:
            observed = self.sensed[pos_request]
        else:
            observed = self.model.disaster_grid[pos_request]
        if human_belief is not None:
            # If the difference is more than 1, return a weighted average.
            if abs(observed - human_belief) > 1:
                corrected = round((observed + human_belief) / 2)
            else:
                corrected = observed
            return corrected
        return observed

    def step(self):
        # Re-sense a new subset of cells each tick.
        self.sense_environment()


#############################
# Run Model and Plot Results
#############################
if __name__ == "__main__":
    # Define simulation parameters.
    share_exploitative = 0.5
    share_of_disaster = 0.2
    initial_trust = 0.5
    number_of_humans = 50

    model = DisasterModel(share_exploitative, share_of_disaster, initial_trust, number_of_humans)
    ticks = 100

    for i in range(ticks):
        model.step()

    # --- Plot Graphs ---
    ticks_range = list(range(ticks))
    # Trust evolution: each trust_data entry is (exp_human, exp_ai, expl_human, expl_ai)
    exp_human_trust = [d[0] for d in model.trust_data]
    exp_ai_trust = [d[1] for d in model.trust_data]
    expl_human_trust = [d[2] for d in model.trust_data]
    expl_ai_trust = [d[3] for d in model.trust_data]

    plt.figure()
    plt.plot(ticks_range, exp_human_trust, label="Exploitative: Human Trust")
    plt.plot(ticks_range, exp_ai_trust, label="Exploitative: AI Trust")
    plt.xlabel("Tick")
    plt.ylabel("Average Trust")
    plt.title("Trust Evolution for Exploitative Agents")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ticks_range, expl_human_trust, label="Exploratory: Human Trust")
    plt.plot(ticks_range, expl_ai_trust, label="Exploratory: AI Trust")
    plt.xlabel("Tick")
    plt.ylabel("Average Trust")
    plt.title("Trust Evolution for Exploratory Agents")
    plt.legend()
    plt.show()

    # Plot information request calls.
    # calls_data entries: (calls_human_exp, calls_ai_exp, calls_human_expl, calls_ai_expl)
    calls_human_exploit = [d[0] for d in model.calls_data]
    calls_ai_exploit = [d[1] for d in model.calls_data]
    calls_human_explor = [d[2] for d in model.calls_data]
    calls_ai_explor = [d[3] for d in model.calls_data]

    plt.figure()
    plt.plot(ticks_range, calls_human_exploit, label="Exploitative: Calls to Humans")
    plt.plot(ticks_range, calls_ai_exploit, label="Exploitative: Calls to AI")
    plt.xlabel("Tick")
    plt.ylabel("Number of Calls")
    plt.title("Information Requests for Exploitative Agents")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ticks_range, calls_human_explor, label="Exploratory: Calls to Humans")
    plt.plot(ticks_range, calls_ai_explor, label="Exploratory: Calls to AI")
    plt.xlabel("Tick")
    plt.ylabel("Number of Calls")
    plt.title("Information Requests for Exploratory Agents")
    plt.legend()
    plt.show()

    # Plot total rewards per tick.
    plt.figure()
    plt.plot(ticks_range, model.rewards_data, label="Total Rewards")
    plt.xlabel("Tick")
    plt.ylabel("Rewards")
    plt.title("Total Rewards per Tick")
    plt.legend()
    plt.show()
