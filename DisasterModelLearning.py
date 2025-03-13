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
      - Data is collected on tokens received (successful actions) and on the number of info calls made to humans vs. AI.
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

        # Statistics counters for visualization (reset each step)
        self.tokens_received_current = 0
        self.info_calls_human_current = 0
        self.info_calls_ai_current = 0

        # Time series for visualization
        self.tokens_received_over_time = []
        self.info_calls_human_over_time = []
        self.info_calls_ai_over_time = []

        # Create the disaster environment.
        # Initially, we set up a decay function from a disaster origin but then adjust so that affected cells (>=3)
        # are no more than 25% of the grid.
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
        # For each human, the direct social neighbours (edges) are considered "closest" (trust=0.7)
        self.social_graph = nx.watts_strogatz_graph(num_humans, k=4, p=0.3)
        for human in self.human_agents:
            # Get friend IDs from the graph.
            human.friends = list(self.social_graph.neighbors(human.unique_id))
            # Initialize trust: 0.7 for direct social neighbours
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
                # Count token only if a reward was earned.
                if reward > 0:
                    self.tokens_received_current += 1
                self.reward_queue.remove(event)

        # Update the disaster environment.
        self.update_environment()

        # Let all agents take their step.
        self.schedule.step()

        # Record the per-step counts.
        self.tokens_received_over_time.append(self.tokens_received_current)
        self.info_calls_human_over_time.append(self.info_calls_human_current)
        self.info_cal
