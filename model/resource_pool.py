import numpy as np


class ResourcePool:

    def __init__(self, total_supply):
        self.total_supply = total_supply
        self.available_supply = total_supply

    def allocate(self, agents):

        total_demand = np.sum([agent.demand for agent in agents])

        # Case 1: Enough supply
        if total_demand <= self.total_supply:
            for agent in agents:
                agent.allocated = agent.demand

        # Case 2: Scarcity → proportional allocation
        else:
            for agent in agents:
                if total_demand > 0:
                    share = agent.demand / total_demand
                    agent.allocated = share * self.total_supply
                else:
                    agent.allocated = 0