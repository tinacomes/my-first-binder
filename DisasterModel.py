import math
import random
import networkx as nx
import matplotlib.pyplot as plt

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import BaseScheduler


class DisasterModel(Model):
    """
    Disaster response model with:
      - A 50x50 grid environment that simulates a disaster.
      - 75 human agents and one (or more) AI agent.
      - The disaster zone is maintained so that no more than 25% of cells are affected (destruction >=3),
        and the environment may improve over time.
      - Data is collected on:
         a) Average trust (humans vs AI) over time,
         b) Reward per timestep,
         c) Interactions (info calls to humans vs. AI) over time.
    """
    def __init__(self, width=50, height=50, num_humans=75, num_ai=1, noise=0.1):
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.num_ai = num_ai
        self.noise = noise  # chance of noise in human info

        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus=False)
        self.current_step = 0

        # Statistics counters for the current step
        self.tokens_received_current = 0
        self.info_calls_human_current = 0
        self.info_calls_ai_current = 0

        # Time series for visualization
        self.tokens_received_over_time = []    # Reward per timestep
        self.info_calls_human_over_time = []     # Info calls to human sources per timestep
        self.info_calls_ai_over_time = []        # Info calls to AI per timestep
        self.avg_trust_human_over_time = []      # Average trust among human agents for human sources
        self.avg_trust_ai_over_time = []         # Average trust among human agents for AI sources

        # Create the disaster environment.
        self.environment = {}
        self._initialize_environment()

        # Create human agents.
        self.human_agents = []
        for i in range(num_humans):
            human = HumanAgent(i, self)
            self.schedule.add(human)
            self.human_agents.append(human)
            pos = (self.random.randrange(width), self.random.randrange(height))
            self.grid.place_agent(human, pos)
            # Initialize each agent’s memory with the true state of its starting cell.
            human.memory[pos] = self.environment[pos]

        # Create AI agents.
        self.ai_agents = []
        for i in range(num_ai):
            ai = AIAgent(num_humans + i, self)
            self.schedule.add(ai)
            self.ai_agents.append(ai)
            pos = (self.random.randrange(width), self.random.randrange(height))
            self.grid.place_agent(ai, pos)

        # Build a social network using a Watts–Strogatz small-world graph.
        self.social_graph = nx.watts_strogatz_graph(num_humans, k=4, p=0.3)
        for human in self.human_agents:
            # Get friend IDs from the graph.
            human.friends = list(self.social_graph.neighbors(human.unique_id))
            # Initialize trust: 0.7 for direct social neighbours.
            for friend_id in human.friends:
                human.trust[friend_id] = 0.7
            # For AI agents (and any non-friend humans, if selected), default trust = 0.5.
            for ai in self.ai_agents:
                human.trust[ai.unique_id] = 0.5

        # Reward queue for delayed rewards.
        # Each event is a tuple: (due_step, agent, target_cell, info_received)
        self.reward_queue = []

    def _initialize_environment(self):
        """Initialize the disaster grid using a decay function and then adjust if too many cells are affected."""
        origin_x = self.random.randrange(self.width)
        origin_y = self.random.randrange(self.height)
        radius = 15.0  # affect cells within approx. 15 units
        for x in range(self.width):
            for y in range(self.height):
                distance = math.sqrt((x - origin_x) ** 2 + (y - origin_y) ** 2)
                # Decay: center gets 5, decaying to 0 at distance==radius.
                level = max(0, 5 - int((distance / radius) * 5))
                self.environment[(x, y)] = level
        self._enforce_max_affected()

    def _enforce_max_affected(self):
        """Ensure that no more than 25% of cells have destruction level >=3."""
        total_cells = self.width * self.height
        max_affected = int(0.25 * total_cells)
        affected_cells = [cell for cell, level in self.environment.items() if level >= 3]
        if len(affected_cells) > max_affected:
            # Randomly reduce some affected cells to level 2.
            to_reduce = len(affected_cells) - max_affected
            cells_to_reduce = random.sample(affected_cells, to_reduce)
            for cell in cells_to_reduce:
                self.environment[cell] = 2

    def update_environment(self):
        """
        At each step, some cells may improve (simulate recovery).
        Then, enforce that the affected area (cells with level >= 3) remains ≤ 25% of the grid.
        """
        for cell, level in self.environment.items():
            # With a small probability, a cell recovers one level.
            if level > 0 and self.random.random() < 0.05:
                self.environment[cell] = level - 1
        self._enforce_max_affected()

    def step(self):
        """
        Advance the model one step:
         1. Process any delayed reward events.
         2. Update the disaster environment.
         3. Let all agents act.
         4. Record statistics.
        """
        # Reset per-step counters.
        self.tokens_received_current = 0
        self.info_calls_human_current = 0
        self.info_calls_ai_current = 0

        # Process delayed reward events.
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
                # Update the agent's memory for the target cell.
                agent.memory[target_cell] = actual

                # Update trust based on info responses.
                for source_id, reported in info_received.items():
                    if reported is None:
                        agent.trust[source_id] = max(0, agent.trust.get(source_id, 0.5) - 0.05)
                    else:
                        if reported != actual:
                            agent.trust[source_id] = max(0, agent.trust.get(source_id, 0.5) - 0.1)
                # Use the delayed outcome to update Q-values via reinforcement learning.
                if hasattr(agent, "update_q_values"):
                    agent.update_q_values(info_received, actual)
                if reward > 0:
                    self.tokens_received_current += reward
                self.reward_queue.remove(event)

        # Update the disaster environment.
        self.update_environment()

        # Let all agents take their step.
        self.schedule.step()

        # After all agents have acted, record average trust levels.
        human_trust_vals = []
        ai_trust_vals = []
        for human in self.human_agents:
            # For human sources, use keys that are in the friend list.
            friend_trusts = [human.trust[f] for f in human.friends if f in human.trust]
            if friend_trusts:
                human_trust_vals.append(sum(friend_trusts) / len(friend_trusts))
            # For AI sources, use keys >= num_humans.
            ai_trusts = [val for key, val in human.trust.items() if key >= self.num_humans]
            if ai_trusts:
                ai_trust_vals.append(sum(ai_trusts) / len(ai_trusts))
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
      - Local sensing of the environment.
      - A memory of past observations.
      - A trust dictionary for friends (social neighbours and AI).
      - A Q-learning mechanism for selecting which source to ask for information.
      - The ability to request up to 3 information sources using an epsilon-greedy rule.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.memory = {}   # cell -> expected destruction level.
        self.trust = {}    # source_id -> trust value.
        self.friends = []  # List of friend (social neighbour) IDs.
        self.score = 0

        # For RL on information acquisition.
        self.q_values = {}     # source_id -> learned Q-value.
        self.epsilon = 0.1     # exploration probability
        self.learning_rate = 0.1  # Q-learning alpha
        self.lambda_weight = 0.5  # weight for mixing trust vs. Q-value

    def step(self):
        # 1. Sense the immediate environment (neighbors of current cell).
        x, y = self.pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.model.width and 0 <= ny < self.model.height:
                    self.memory[(nx, ny)] = self.model.environment[(nx, ny)]

        # 2. Select a target cell—the one with the highest expected destruction.
        if self.memory:
            target_cell = max(self.memory.items(), key=lambda item: item[1])[0]
        else:
            target_cell = self.pos

        # 3. Decide from whom to request information.
        # Potential sources include direct social neighbours (friends) and AI agents.
        potential_sources = []
        potential_sources.extend(self.friends)
        for ai in self.model.ai_agents:
            potential_sources.append(ai.unique_id)
        potential_sources = list(set(potential_sources))

        # Compute expected value for each source as a weighted combination of trust and Q-value.
        expected_values = {
            src: self.lambda_weight * self.trust.get(src, 0.5) + (1 - self.lambda_weight) * self.q_values.get(src, 0)
            for src in potential_sources
        }

        # Select up to 3 sources using an epsilon-greedy strategy.
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
        # 4. Request information from the selected sources.
        for source_id in selected_sources:
            source_agent = self._find_agent_by_id(source_id)
            if source_agent:
                if isinstance(source_agent, HumanAgent):
                    self.model.info_calls_human_current += 1
                elif isinstance(source_agent, AIAgent):
                    self.model.info_calls_ai_current += 1

                reported = source_agent.provide_info(target_cell, requester=self)
                # Add noise for human sources.
                if isinstance(source_agent, HumanAgent) and reported is not None:
                    if self.model.random.random() < self.model.noise:
                        reported = self.model.random.randrange(6)
                info_received[source_id] = reported
            else:
                info_received[source_id] = None

        # 5. Integrate received info with current memory.
        values = []
        if target_cell in self.memory:
            values.append(self.memory[target_cell])
        for rep in info_received.values():
            if rep is not None:
                values.append(rep)
        if values:
            new_belief = int(round(sum(values) / len(values)))
            self.memory[target_cell] = new_belief

        # 6. Decide where to send a token.
        chosen_cell = max(self.memory.items(), key=lambda item: item[1])[0]

        # 7. Schedule a delayed reward event (2 steps later) with the chosen cell and the info received.
        due_step = self.model.current_step + 2
        self.model.reward_queue.append((due_step, self, chosen_cell, info_received))

    def update_q_values(self, info_received, actual):
        """
        Update the Q-values for each source based on the delayed outcome:
          - +1 if the reported info matches the actual destruction level.
          - -1 if the reported info is wrong.
          - -0.5 if no info was received.
        """
        for source_id, reported in info_received.items():
            if reported is None:
                reward_signal = -0.5
            elif reported == actual:
                reward_signal = 1
            else:
                reward_signal = -1
            old_q = self.q_values.get(source_id, 0)
            new_q = old_q + self.learning_rate * (reward_signal - old_q)
            self.q_values[source_id] = new_q

    def provide_info(self, cell, requester=None):
        """
        Provide information about a given cell.
        If this human agent is in a cell with destruction level >= 3, they cannot respond.
        Otherwise, return the known value.
        """
        if self.model.environment[self.pos] >= 3:
            return None
        return self.memory.get(cell, self.model.environment[cell])

    def _find_agent_by_id(self, agent_id):
        """Helper: search for an agent by its unique_id in the model schedule."""
        for agent in self.model.schedule.agents:
            if agent.unique_id == agent_id:
                return agent
        return None


class AIAgent(Agent):
    """
    AI agent that:
      - Always responds to information requests.
      - Maintains an overview of 10% of grid cells (knows their true destruction level).
      - When requested, with 50% chance may return the requester’s current belief instead of the true value.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        total_cells = model.width * model.height
        num_known = int(0.1 * total_cells)
        all_cells = list(model.environment.keys())
        self.overview = set(self.random.sample(all_cells, num_known))

    def step(self):
        # AI agents do not take independent actions in this model.
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
    # Run the model for a number of steps and then plot the metrics.
    model = DisasterModel()
    steps = 50

    for i in range(steps):
        model.step()
        avg_score = sum([agent.score for agent in model.human_agents]) / len(model.human_agents)
        print(f"Step {i}: Average human agent score: {avg_score:.2f}")

    time_steps = list(range(steps))

    # Plot (a): Average trust evolution for human vs AI.
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, model.avg_trust_human_over_time, label="Avg Trust in Humans")
    plt.plot(time_steps, model.avg_trust_ai_over_time, label="Avg Trust in AI")
    plt.xlabel("Time Step")
    plt.ylabel("Average Trust")
    plt.title("Evolution of Average Trust (Humans vs. AI)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot (b): Reward per timestep.
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, model.tokens_received_over_time, label="Reward per Timestep")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.title("Reward per Timestep")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot (c): Interactions (info calls) between humans and AI.
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, model.info_calls_human_over_time, label="Info Calls to Humans")
    plt.plot(time_steps, model.info_calls_ai_over_time, label="Info Calls to AI")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Info Calls")
    plt.title("Information Requests: Humans vs. AI")
    plt.legend()
    plt.tight_layout()
    plt.show()
