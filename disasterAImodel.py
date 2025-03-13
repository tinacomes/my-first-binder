import random
import numpy as np
import networkx as nx
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

############################
# The Disaster Model
############################

class DisasterModel(Model):
    def __init__(self, width=50, height=50, num_humans=75, num_ai=2, noise=0.1):
        self.width = width
        self.height = height
        self.num_humans = num_humans
        self.num_ai = num_ai
        self.noise = noise
        
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=False)
        # The environment stores the disaster destruction levels (0-5)
        self.environment = np.zeros((width, height), dtype=int)
        
        # Initialize the disaster pattern using a decay function
        self.init_disaster()
        
        # Create human agents
        self.humans = []
        for i in range(num_humans):
            pos = (random.randrange(width), random.randrange(height))
            human = HumanAgent(i, self, pos)
            self.humans.append(human)
            self.schedule.add(human)
            self.grid.place_agent(human, pos)
            
        # Create AI agents
        self.ai_agents = []
        for i in range(num_ai):
            ai = AIAgent(num_humans + i, self)
            self.ai_agents.append(ai)
            self.schedule.add(ai)
            
        # Build a social network among humans (using a Watts-Strogatz small-world network)
        self.create_social_network()
        
        self.current_step = 0

    def init_disaster(self):
        # Choose an epicenter at random
        epicenter = (random.randrange(self.width), random.randrange(self.height))
        decay_factor = 5  # determines how quickly destruction decays with distance
        
        # Assign destruction levels based on Manhattan distance from the epicenter
        for x in range(self.width):
            for y in range(self.height):
                dist = abs(x - epicenter[0]) + abs(y - epicenter[1])
                level = max(5 - (dist // decay_factor), 0)
                self.environment[x, y] = level

    def create_social_network(self):
        # Build a Watts-Strogatz small-world network for the human agents
        G = nx.watts_strogatz_graph(self.num_humans, k=4, p=0.3)
        for human in self.humans:
            # Save neighbor IDs (each human is identified by its unique_id)
            human.social_neighbors = list(G.neighbors(human.unique_id))

    def update_environment(self):
        # A simple dynamic: cells near a maximum destruction cell (level 5) intensify.
        new_env = self.environment.copy()
        for x in range(self.width):
            for y in range(self.height):
                # Get the neighborhood of cell (x,y)
                neighbors = self.grid.get_neighborhood((x, y), moore=True, include_center=False)
                if any(self.environment[nx, ny] == 5 for nx, ny in neighbors):
                    if new_env[x, y] < 5:
                        new_env[x, y] += 1
        self.environment = new_env

    def step(self):
        self.update_environment()
        self.schedule.step()
        self.current_step += 1

############################
# The Human Agent
############################

class HumanAgent(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        # Memory of observed states: keys are cell coordinates, values are lists of observed destruction levels.
        self.memory = {}
        # Expectation: for every cell, what the agent believes its destruction level is.
        self.expectation = {(x, y): 0 for x in range(model.width) for y in range(model.height)}
        # Initialize expectation of current cell from the actual environment.
        self.expectation[pos] = model.environment[pos[0], pos[1]]
        # Trust levels for information sources:
        # For humans, initial trust is 0.7; for AI, 0.5.
        self.trust = {}
        self.pending_rewards = []  # list of tuples: (cell, decision_round, expected_level)
        self.noise = model.noise
        # social_neighbors will be set by the model's social network generator
        self.social_neighbors = []

    def step(self):
        # 1. Sense the environment in the neighborhood.
        self.sense_environment()
        # 2. Request information from up to 3 social neighbors and 1 AI.
        self.request_information()
        # 3. Decide which cell is most affected and send a token.
        target = self.decide_target()
        self.pending_rewards.append((target, self.model.current_step, self.expectation[target]))
        # 4. Process any pending rewards (with a delay of 2 time steps).
        self.process_pending_rewards()

    def sense_environment(self):
        # Look at all cells in the Moore neighborhood (including own cell)
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        for cell in neighbors:
            x, y = cell
            observed = self.model.environment[x, y]
            self.memory.setdefault(cell, []).append(observed)
            # For simplicity, update expectation directly with the sensed value.
            self.expectation[cell] = observed

    def request_information(self):
        sources = []
        # Request from up to 3 human neighbors (if available)
        if self.social_neighbors:
            random.shuffle(self.social_neighbors)
            sources.extend(self.social_neighbors[:3])
        # Also request from one random AI agent (if available)
        if self.model.ai_agents:
            ai_source = random.choice(self.model.ai_agents)
            sources.append(ai_source.unique_id)
            
        for source_id in sources:
            source_agent = self.get_agent_by_id(source_id)
            if source_agent is not None:
                # If the source is a human agent, call share_info; if AI, call respond_info.
                if isinstance(source_agent, HumanAgent):
                    info = source_agent.share_info()
                    # A human in a highly destroyed cell (>= 3) cannot respond.
                    if self.model.environment[source_agent.pos[0], source_agent.pos[1]] >= 3:
                        info = None
                    # With probability 'noise', return an incorrect (random) value.
                    if info is not None and random.random() < self.noise:
                        info = (info[0], random.randint(0, 5))
                    if info is not None:
                        cell, destruction = info
                        trust_level = self.trust.get(source_id, 0.7)
                        # Update the expectation for the cell using a weighted average.
                        self.expectation[cell] = (self.expectation.get(cell, 0) + trust_level * destruction) / (1 + trust_level)
                    else:
                        # No information received; reduce trust slightly.
                        self.trust[source_id] = self.trust.get(source_id, 0.7) - 0.05
                else:  # AI agent
                    info = source_agent.respond_info(self)
                    if info is not None:
                        cell, destruction = info
                        trust_level = self.trust.get(source_id, 0.5)
                        self.expectation[cell] = (self.expectation.get(cell, 0) + trust_level * destruction) / (1 + trust_level)

    def share_info(self):
        # When requested, share info about the current cell.
        if self.model.environment[self.pos[0], self.pos[1]] >= 3:
            return None
        else:
            return (self.pos, self.model.environment[self.pos[0], self.pos[1]])

    def decide_target(self):
        # Choose the cell with the highest expected destruction.
        target = max(self.expectation.items(), key=lambda x: x[1])[0]
        return target

    def process_pending_rewards(self):
        new_pending = []
        for (cell, decision_round, expected) in self.pending_rewards:
            if self.model.current_step - decision_round >= 2:
                actual = self.model.environment[cell[0], cell[1]]
                reward = 0
                if actual == 4:
                    reward = 1
                elif actual == 5:
                    reward = 3
                # Update the expectation for that cell towards the actual value.
                self.expectation[cell] = (self.expectation[cell] + actual) / 2
                # (Further trust updates could be done here based on whether the information proved wrong.)
            else:
                new_pending.append((cell, decision_round, expected))
        self.pending_rewards = new_pending

    def get_agent_by_id(self, agent_id):
        # Look up an agent among humans and AI based on unique_id.
        for human in self.model.humans:
            if human.unique_id == agent_id:
                return human
        for ai in self.model.ai_agents:
            if ai.unique_id == agent_id:
                return ai
        return None

############################
# The AI Agent
############################

class AIAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # AI's coverage: randomly select 10% of all cells in the grid.
        total_cells = model.width * model.height
        num_cells = int(total_cells * 0.1)
        self.coverage = random.sample([(x, y) for x in range(model.width) for y in range(model.height)], num_cells)

    def step(self):
        # Optionally update coverage dynamically. For now, we keep it static.
        pass

    def respond_info(self, requester):
        # With 50% chance, confirm the requester's own prior belief.
        if random.random() < 0.5:
            cell = max(requester.expectation.items(), key=lambda x: x[1])[0]
            destruction = requester.expectation[cell]
            return (cell, destruction)
        else:
            # Otherwise, from the AI's coverage, return the cell with the highest actual destruction.
            best_cell = None
            best_value = -1
            for cell in self.coverage:
                x, y = cell
                value = self.model.environment[x, y]
                if value > best_value:
                    best_value = value
                    best_cell = cell
            return (best_cell, best_value)

############################
# Running the Model
############################

if __name__ == "__main__":
    # Create the model and run for a few steps
    model = DisasterModel()
    for i in range(10):
        model.step()
        print(f"Step {i+1} completed.")
    
    # For demonstration, print the current disaster environment (a 50x50 grid)
    print("Final disaster environment (grid of destruction levels):")
    print(model.environment)
