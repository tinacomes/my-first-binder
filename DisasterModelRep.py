import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import warnings

# Suppress deprecation warnings (if desired)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler

class DisasterModel(Model):
    """
    Disaster response model with:
      - A 50x50 grid disaster environment.
      - 75 human agents and one (or more) AI agent.
      - Disaster dynamics that limit affected cells.
      - Social network with trust initialized based on network distance.
      - A reputation system and RL for information acquisition.
      - Two groups of human agents: exploratory and exploitative.
    """
    def __init__(self, width=50, height=50, num_humans=75, num_ai=1, noise=0.1):
        super().__init__()  # needed so that self.random is defined.
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.num_ai = num_ai
        self.noise = noise  # chance of noise in human info
        
        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.current_step = 0

        # Counters for per–step statistics.
        self.tokens_received_current = 0
        self.info_calls_human_current = 0
        self.info_calls_ai_current = 0

        # Time series for visualization.
        self.tokens_received_over_time = []    
        self.info_calls_human_over_time = []     
        self.info_calls_ai_over_time = []        
        self.avg_trust_human_over_time = []      
        self.avg_trust_ai_over_time = []         

        # Create the disaster environment.
        self.environment = {}
        self._initialize_environment()

        # Create human agents.
        self.human_agents = []
        for i in range(num_humans):
            # Split agents into two groups: first half exploratory, second half exploitative.
            strategy = "exploratory" if i < num_humans // 2 else "exploitative"
            human = HumanAgent(i, self, strategy=strategy)
            self.schedule.add(human)
            self.human_agents.append(human)
            pos = (self.random.randrange(width), self.random.randrange(height))
            self.grid.place_agent(human, pos)
            # Initialize each agent's memory with the true state of its starting cell.
            human.memory[pos] = self.environment[pos]

        # Create AI agents.
        self.ai_agents = []
        for i in range(num_ai):
            ai = AIAgent(num_humans + i, self)
            self.schedule.add(ai)
            self.ai_agents.append(ai)
            pos = (self.random.randrange(width), self.random.randrange(height))
            self.grid.place_agent(ai, pos)

        # Build a social network among human agents.
        # You can change the network topology here (e.g. using nx.watts_strogatz_graph or nx.barabasi_albert_graph).
        self.social_graph = nx.watts_strogatz_graph(num_humans, k=4, p=0.3)
        for human in self.human_agents:
            # Determine direct neighbors from the graph.
            human.friends = list(self.social_graph.neighbors(human.unique_id))
        
        # Initialize trust based on network distance.
        for human in self.human_agents:
            for other in self.social_graph.nodes():
                if other == human.unique_id:
                    continue
                try:
                    d = nx.shortest_path_length(self.social_graph, source=human.unique_id, target=other)
                except nx.NetworkXNoPath:
                    d = None
                if d is not None:
                    if d == 1:
                        trust_value = 0.7
                    elif d == 2:
                        trust_value = 0.5
                    elif d == 3:
                        trust_value = 0.3
                    else:
                        trust_value = 0.2
                else:
                    trust_value = 0.2
                human.trust[other] = trust_value
                # Also initialize reputation for other human agents.
                human.reputation[other] = 0.5
            # For AI agents, default trust and reputation is set to 0.5.
            for ai in self.ai_agents:
                human.trust[ai.unique_id] = 0.5
                human.reputation[ai.unique_id] = 0.5

        # Reward queue for delayed rewards.
        # Each event: (due_step, agent, target_cell, info_received)
        self.reward_queue = []

    def _initialize_environment(self):
        """Initialize disaster grid using a decay function; then enforce that affected cells are ≤25%."""
        origin_x = self.random.randrange(self.width)
        origin_y = self.random.randrange(self.height)
        radius = 15.0  # affect cells within approx. 15 units.
        for x in range(self.width):
            for y in range(self.height):
                distance = math.sqrt((x - origin_x) ** 2 + (y - origin_y) ** 2)
                level = max(0, 5 - int((distance / radius) * 5))
                self.environment[(x, y)] = level
        self._enforce_max_affected()

    def _enforce_max_affected(self):
        """Ensure no more than 25% of cells have destruction level >=3."""
        total_cells = self.width * self.height
        max_affected = int(0.25 * total_cells)
        affected_cells = [cell for cell, level in self.environment.items() if level >= 3]
        if len(affected_cells) > max_affected:
            to_reduce = len(affected_cells) - max_affected
            cells_to_reduce = random.sample(affected_cells, to_reduce)
            for cell in cells_to_reduce:
                self.environment[cell] = 2

    def update_environment(self):
        """
        Some cells may recover over time. Then enforce affected area remains ≤25%.
        """
        for cell, level in self.environment.items():
            if level > 0 and self.random.random() < 0.05:
                self.environment[cell] = level - 1
        self._enforce_max_affected()

    def step(self):
        """
        One model step:
         1. Process delayed reward events.
         2. Update disaster environment.
         3. Let all agents act.
         4. Record statistics.
        """
        # Reset per-step counters.
        self.tokens_received_current = 0
        self.info_calls_human_current = 0
        self.info_calls_ai_current = 0

        # Process reward events.
        for event in self.reward_queue[:]:
            due, agent, target_cell, info_received = event
            if due <= self.current_step:
                actual = self.environment[target_cell]
                reward = 0
                if actual == 4:
                    reward = 1
                elif actual == 5:
                    reward = 3
                agent.score += reward
                agent.memory[target_cell] = actual

                # Update trust and reputation based on info responses.
                for source_id, reported in info_received.items():
                    if reported is None:
                        agent.trust[source_id] = max(0, agent.trust.get(source_id, 0.5) - 0.05)
                    else:
                        if reported != actual:
                            agent.trust[source_id] = max(0, agent.trust.get(source_id, 0.5) - 0.1)
                if hasattr(agent, "update_q_values"):
                    agent.update_q_values(info_received, actual)
                if reward > 0:
                    self.tokens_received_current += reward
                self.reward_queue.remove(event)

        self.update_environment()
        self.schedule.step()

        # Record average trust (separately for human sources and AI sources).
        human_trust_vals = []
        ai_trust_vals = []
        for human in self.human_agents:
            friend_trusts = [human.trust[f] for f in human.friends if f in human.trust]
            if friend_trusts:
                human_trust_vals.append(sum(friend_trusts) / len(friend_trusts))
            ai_vals = [val for key, val in human.trust.items() if key >= self.num_humans]
            if ai_vals:
                ai_trust_vals.append(sum(ai_vals) / len(ai_vals))
        avg_trust_humans = sum(human_trust_vals) / len(human_trust_vals) if human_trust_vals else 0
        avg_trust_ai = sum(ai_trust_vals) / len(ai_trust_vals) if ai_trust_vals else 0

        self.avg_trust_human_over_time.append(avg_trust_humans)
        self.avg_trust_ai_over_time.append(avg_trust_ai)

        # Record per-step counts.
        self.tokens_received_over_time.append(self.tokens_received_current)
        self.info_calls_human_over_time.append(self.info_calls_human_current)
        self.info_calls_ai_over_time.append(self.info_calls_ai_current)

        self.current_step += 1


class HumanAgent(Agent):
    """
    Human agent with:
      - Local sensing and memory.
      - Trust and reputation dictionaries for sources.
      - A reinforcement learning mechanism for selecting sources.
      - Two strategies: exploratory vs. exploitative.
         * Exploratory: higher exploration, accepts all info.
         * Exploitative: lower exploration, applies confirmation bias.
    """
    def __init__(self, unique_id, model, strategy="exploratory"):
        super().__init__(model)
        self.unique_id = unique_id
        self.strategy = strategy  # "exploratory" or "exploitative"
        self.epsilon = 0.2 if self.strategy == "exploratory" else 0.05
        self.memory = {}      # cell -> expected destruction level.
        self.trust = {}       # source_id -> trust value.
        self.reputation = {}  # source_id -> reputation value.
        self.friends = []     # List of direct social neighbors.
        self.score = 0

        # RL parameters.
        self.q_values = {}     # source_id -> learned Q-value.
        self.learning_rate = 0.1
        # We'll combine trust, Q-value, and reputation with fixed weights.
        self.lambda_trust = 0.4
        self.lambda_q = 0.3
        self.lambda_rep = 0.3

    def step(self):
        # 1. Sense the environment around current cell.
        x, y = self.pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < self.model.width and 0 <= ny_ < self.model.height:
                    self.memory[(nx_, ny_)] = self.model.environment[(nx_, ny_)]

        # 2. Choose a target cell (cell with highest expected destruction).
        if self.memory:
            target_cell = max(self.memory.items(), key=lambda item: item[1])[0]
        else:
            target_cell = self.pos

        # 3. Decide which sources to query.
        potential_sources = set(self.friends)
        for ai in self.model.ai_agents:
            potential_sources.add(ai.unique_id)
        potential_sources = list(potential_sources)
        expected_values = {
            src: (self.lambda_trust * self.trust.get(src, 0.5) +
                  self.lambda_q * self.q_values.get(src, 0) +
                  self.lambda_rep * self.reputation.get(src, 0.5))
            for src in potential_sources
        }

        selected_sources = []
        available_sources = potential_sources.copy()
        for _ in range(min(3, len(available_sources))):
            if random.random() < self.epsilon:
                chosen = random.choice(available_sources)
            else:
                chosen = max(available_sources, key=lambda s: expected_values[s])
            selected_sources.append(chosen)
            available_sources.remove(chosen)

        info_received = {}
        # 4. Request info from selected sources.
        for source_id in selected_sources:
            source_agent = self._find_agent_by_id(source_id)
            if source_agent:
                if isinstance(source_agent, HumanAgent):
                    self.model.info_calls_human_current += 1
                elif isinstance(source_agent, AIAgent):
                    self.model.info_calls_ai_current += 1

                reported = source_agent.provide_info(target_cell, requester=self)
                if isinstance(source_agent, HumanAgent) and reported is not None:
                    if self.model.random.random() < self.model.noise:
                        reported = self.model.random.randrange(6)
                info_received[source_id] = reported
            else:
                info_received[source_id] = None

        # 5. Integrate received info with current memory.
        # Apply confirmation bias for exploitative agents: only accept info within a threshold of current belief.
        current_belief = self.memory.get(target_cell, self.model.environment[target_cell])
        values = [current_belief]
        for src, rep_val in info_received.items():
            if rep_val is not None:
                if self.strategy == "exploitative":
                    if abs(rep_val - current_belief) <= 1:
                        values.append(rep_val)
                    # Otherwise, ignore info that deviates too much.
                else:
                    values.append(rep_val)
        new_belief = int(round(sum(values) / len(values)))
        self.memory[target_cell] = new_belief

        # 6. Decide where to send a token.
        chosen_cell = max(self.memory.items(), key=lambda item: item[1])[0]

        # 7. Schedule a delayed reward event (2 steps later) with the chosen cell and info received.
        due_step = self.model.current_step + 2
        self.model.reward_queue.append((due_step, self, chosen_cell, info_received))

    def update_q_values(self, info_received, actual):
        """
        Update Q-values and reputation for each source:
         - +1 (or +0.1 to reputation) if info is correct.
         - -1 (or -0.1) if info is wrong.
         - -0.5 (or -0.05) if no info.
        """
        for source_id, reported in info_received.items():
            if reported is None:
                reward_signal = -0.5
                rep_update = -0.05
            elif reported == actual:
                reward_signal = 1
                rep_update = 0.1
            else:
                reward_signal = -1
                rep_update = -0.1

            old_q = self.q_values.get(source_id, 0)
            self.q_values[source_id] = old_q + self.learning_rate * (reward_signal - old_q)

            old_rep = self.reputation.get(source_id, 0.5)
            new_rep = max(0, min(1, old_rep + rep_update))
            self.reputation[source_id] = new_rep

    def provide_info(self, cell, requester=None):
        """
        Provide info about a given cell.
        If this agent is in a highly affected cell (destruction level >=3), it cannot respond.
        """
        if self.model.environment[self.pos] >= 3:
            return None
        return self.memory.get(cell, self.model.environment[cell])

    def _find_agent_by_id(self, agent_id):
        """Return the agent with the given unique_id from the model's schedule."""
        for agent in self.model.schedule.agents:
            if agent.unique_id == agent_id:
                return agent
        return None


class AIAgent(Agent):
    """
    AI agent that:
      - Always responds.
      - Has an overview of 10% of grid cells.
      - With 50% chance may return the requester's belief rather than the true value.
    """
    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
        total_cells = model.width * model.height
        num_known = int(0.1 * total_cells)
        all_cells = list(model.environment.keys())
        self.overview = set(self.random.sample(all_cells, num_known))

    def step(self):
        pass

    def provide_info(self, cell, requester=None):
        if cell in self.overview:
            if self.model.random.random() < 0.5:
                if requester is not None and cell in requester.memory:
                    return requester.memory[cell]
                else:
                    return self.model.environment[cell]
            else:
                return self.model.environment[cell]
        else:
            if requester is not None and cell in requester.memory:
                return requester.memory[cell]
            else:
                return self.model.environment[cell]


if __name__ == "__main__":
    # Run the model for a number of steps and plot the metrics.
    model = DisasterModel()
    steps = 50

    for i in range(steps):
        model.step()
        avg_score = sum([agent.score for agent in model.human_agents]) / len(model.human_agents)
        print(f"Step {i}: Average human agent score: {avg_score:.2f}")

    time_steps = list(range(steps))

    # (a) Plot evolution of average trust for human vs. AI.
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, model.avg_trust_human_over_time, label="Avg Trust in Humans")
    plt.plot(time_steps, model.avg_trust_ai_over_time, label="Avg Trust in AI")
    plt.xlabel("Time Step")
    plt.ylabel("Average Trust")
    plt.title("Evolution of Average Trust")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (b) Plot reward per timestep.
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, model.tokens_received_over_time, label="Reward per Timestep")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Reward per Timestep")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (c) Plot interactions (info calls) between humans and AI.
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, model.info_calls_human_over_time, label="Info Calls to Humans")
    plt.plot(time_steps, model.info_calls_ai_over_time, label="Info Calls to AI")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Info Calls")
    plt.title("Information Requests: Humans vs. AI")
    plt.legend()
    plt.tight_layout()
    plt.show()
